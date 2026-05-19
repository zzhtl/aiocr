#[cfg(aiocr_has_generated_rec)]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
#[cfg(aiocr_has_generated_rec)]
use std::sync::atomic::Ordering;

#[cfg(aiocr_has_generated_rec)]
use aiocr_core::decode::CtcDecoder;
#[cfg(aiocr_has_generated_rec)]
use aiocr_core::preprocess::preprocess_for_recognition;
#[cfg(aiocr_has_generated_rec)]
use image::DynamicImage;

use crate::ai_model::AiModelInfo;
#[cfg(aiocr_has_generated_rec)]
use crate::ai_model::{AiModelManifest, AiModelMetrics};
use crate::config::TrainingConfig;
#[cfg(aiocr_has_generated_rec)]
use crate::data::dataset::OcrDataset;
#[cfg(aiocr_has_generated_rec)]
use crate::data::dataset::OcrItem;
use crate::error::TrainError;

/// 训练进度。
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub loss: f32,
    pub accuracy: f32,
}

/// 训练完成摘要。
#[derive(Debug, Clone)]
pub struct TrainingSummary {
    pub model_dir: PathBuf,
    pub model: AiModelInfo,
}

/// 训练进度回调。
pub trait TrainingCallback: Send {
    fn on_epoch_start(&mut self, epoch: usize, total_epochs: usize);
    fn on_batch_end(&mut self, progress: &TrainingProgress);
    fn on_epoch_end(&mut self, progress: &TrainingProgress);
    fn on_training_complete(&mut self, summary: &TrainingSummary);
}

/// 默认日志回调。
pub struct LogCallback;

impl TrainingCallback for LogCallback {
    fn on_epoch_start(&mut self, epoch: usize, total_epochs: usize) {
        tracing::info!("Epoch {epoch}/{total_epochs} 开始");
    }

    fn on_batch_end(&mut self, progress: &TrainingProgress) {
        tracing::debug!(
            "Epoch {}/{} Batch {}/{} loss={:.4} acc={:.2}%",
            progress.epoch,
            progress.total_epochs,
            progress.batch,
            progress.total_batches,
            progress.loss,
            progress.accuracy * 100.0
        );
    }

    fn on_epoch_end(&mut self, progress: &TrainingProgress) {
        tracing::info!(
            "Epoch {}/{} 完成 loss={:.4} acc={:.2}%",
            progress.epoch,
            progress.total_epochs,
            progress.loss,
            progress.accuracy * 100.0
        );
    }

    fn on_training_complete(&mut self, summary: &TrainingSummary) {
        tracing::info!(
            "训练完成: {} -> {:?}",
            summary.model.name,
            summary.model_dir
        );
    }
}

/// 训练器。
///
/// 当前实现会基于已有识别 AI 模型继续训练识别权重：
/// 1. 读取标注好的文本行数据集
/// 2. 加载当前激活的本地 AI 识别模型权重（为空时使用内嵌默认模型）
/// 3. 使用近似 CTC 的对齐损失继续训练识别网络
/// 4. 导出新的本地 AI 权重包，供 GUI 热切换
pub struct Trainer {
    #[cfg_attr(not(aiocr_has_generated_rec), allow(dead_code))]
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// 开始训练。
    pub fn train(
        &self,
        callback: &mut dyn TrainingCallback,
    ) -> Result<TrainingSummary, TrainError> {
        self.train_with_cancel(callback, None)
    }

    /// 开始训练，支持外部取消。
    pub fn train_with_cancel(
        &self,
        callback: &mut dyn TrainingCallback,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<TrainingSummary, TrainError> {
        #[cfg(aiocr_has_generated_rec)]
        {
            return self.train_generated(callback, cancel_flag);
        }

        #[cfg(not(aiocr_has_generated_rec))]
        {
            let _ = callback;
            let _ = cancel_flag;
            Err(TrainError::Training(
                "当前环境没有可训练的识别 AI 模型，请先确认 models/rec.onnx 已存在并重新构建。"
                    .to_string(),
            ))
        }
    }
}

#[cfg(aiocr_has_generated_rec)]
impl Trainer {
    fn train_generated(
        &self,
        callback: &mut dyn TrainingCallback,
        cancel_flag: Option<&AtomicBool>,
    ) -> Result<TrainingSummary, TrainError> {
        use burn::backend::{Autodiff, NdArray};
        use burn::module::AutodiffModule;
        use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
        use burn::prelude::{Int, Tensor, TensorData};

        type TrainBackend = Autodiff<NdArray<f32>>;
        type InferBackend = NdArray<f32>;
        type TrainModel = aiocr_core::models::rec_generated::Model<TrainBackend>;
        type InferModel = aiocr_core::models::rec_generated::Model<InferBackend>;

        tracing::info!("训练配置: {:?}", self.config);
        std::fs::create_dir_all(&self.config.artifact_dir)?;

        let base_model = resolve_base_model(&self.config)?;
        let decoder = CtcDecoder::from_dict_file(&base_model.dict_path).map_err(|err| {
            TrainError::Dataset(format!(
                "加载基础模型字典 {} 失败: {err}",
                base_model.dict_path.display()
            ))
        })?;
        let char_to_idx = OcrDataset::build_char_map(&base_model.dict_path)?;
        let dataset = OcrDataset::from_dir(&self.config.dataset_path, &char_to_idx)?;
        if dataset.items().is_empty() {
            return Err(TrainError::Dataset(format!(
                "数据集 {:?} 中未找到有效样本，请检查 images/ 和 labels.txt",
                self.config.dataset_path
            )));
        }

        let (train_items, validation_items) = split_dataset(dataset.items());
        let total_batches = train_items
            .len()
            .div_ceil(self.config.batch_size.max(1))
            .max(1);

        let device = Default::default();
        let mut model =
            load_train_model::<TrainBackend>(&device, base_model.weights_path.as_deref())?;
        let mut optimizer = AdamWConfig::new().init::<TrainBackend, TrainModel>();

        let initial_model = model.valid();
        let initial_accuracy = evaluate_accuracy::<InferBackend, InferModel>(
            &initial_model,
            decoder_ref(&decoder),
            &validation_items,
        )?;

        let mut best_accuracy = initial_accuracy;
        let mut best_loss = f32::MAX;
        let mut best_bytes: Option<Vec<u8>> = Some(save_model_bytes(&initial_model)?);
        let mut best_metrics: Option<AiModelMetrics> = Some(AiModelMetrics {
            train_samples: train_items.len(),
            validation_samples: validation_items.len(),
            accuracy: initial_accuracy,
            avg_loss: 0.0,
        });

        for epoch in 1..=self.config.num_epochs.max(1) {
            check_cancel(cancel_flag)?;
            callback.on_epoch_start(epoch, self.config.num_epochs.max(1));

            let mut running_loss = 0.0f32;
            let mut running_correct = 0usize;
            let mut processed = 0usize;

            for (batch_index, batch) in train_items
                .chunks(self.config.batch_size.max(1))
                .enumerate()
            {
                check_cancel(cancel_flag)?;

                let mut batch_loss = 0.0f32;
                let mut batch_correct = 0usize;

                for item in batch {
                    let raw_image = load_image(&item.image_path)?;
                    // 训练时应用数据增强提高泛化能力
                    let image = crate::data::augment::augment(&raw_image);
                    let input_data = preprocess_for_recognition(&image);
                    let width = input_data.len() / (3 * 48);
                    let input = Tensor::<TrainBackend, 4>::from_data(
                        TensorData::new(input_data, [1, 3, 48, width]),
                        &device,
                    );

                    let probabilities = normalize_prob_dims(model.forward(input))?;
                    let [_batch, time_steps, _classes] = probabilities.dims();
                    let aligned_targets = build_aligned_targets(&item.label_indices, time_steps);
                    let indices = Tensor::<TrainBackend, 3, Int>::from_data(
                        TensorData::new(aligned_targets, [1, time_steps, 1]),
                        &device,
                    );
                    let gathered: Tensor<TrainBackend, 2> =
                        probabilities.clone().gather(2, indices).squeeze_dim(2);
                    let loss = gathered.clamp_min(1e-6).log().neg().mean();
                    let loss_value = loss.clone().into_scalar();
                    let prediction = decode_output(decoder_ref(&decoder), probabilities)?;

                    if prediction.0 == item.label_text {
                        batch_correct += 1;
                    }

                    let grads = GradientsParams::from_grads(loss.backward(), &model);
                    model = optimizer.step(self.config.learning_rate, model, grads);
                    batch_loss += loss_value;
                }

                processed += batch.len();
                running_correct += batch_correct;
                running_loss += batch_loss / batch.len().max(1) as f32;

                let progress = TrainingProgress {
                    epoch,
                    total_epochs: self.config.num_epochs.max(1),
                    batch: batch_index + 1,
                    total_batches,
                    loss: running_loss / (batch_index + 1) as f32,
                    accuracy: running_correct as f32 / processed.max(1) as f32,
                };
                callback.on_batch_end(&progress);
            }

            let valid_model = model.valid();
            let accuracy = evaluate_accuracy::<InferBackend, InferModel>(
                &valid_model,
                decoder_ref(&decoder),
                &validation_items,
            )?;
            let avg_loss = if total_batches > 0 {
                running_loss / total_batches as f32
            } else {
                0.0
            };

            let progress = TrainingProgress {
                epoch,
                total_epochs: self.config.num_epochs.max(1),
                batch: total_batches,
                total_batches,
                loss: avg_loss,
                accuracy,
            };
            callback.on_epoch_end(&progress);

            if accuracy > best_accuracy || (accuracy == best_accuracy && avg_loss <= best_loss) {
                best_accuracy = accuracy;
                best_loss = avg_loss;
                best_bytes = Some(save_model_bytes(&valid_model)?);
                best_metrics = Some(AiModelMetrics {
                    train_samples: train_items.len(),
                    validation_samples: validation_items.len(),
                    accuracy,
                    avg_loss,
                });
            }
        }

        let Some(best_bytes) = best_bytes else {
            return Err(TrainError::Training("训练没有产出可用模型".to_string()));
        };
        let Some(metrics) = best_metrics else {
            return Err(TrainError::Training("训练没有产出可用指标".to_string()));
        };

        let model_name = build_model_name(&self.config, &base_model.name);
        let manifest = AiModelManifest::new(model_name, base_model.name, metrics);
        let model_dir = manifest.save_to_root(
            &self.config.artifact_dir,
            &best_bytes,
            &base_model.dict_path,
        )?;
        let model = AiModelInfo::load_from_dir(&model_dir)?;
        let summary = TrainingSummary { model_dir, model };
        callback.on_training_complete(&summary);
        Ok(summary)
    }
}

#[cfg(aiocr_has_generated_rec)]
struct BaseModelSpec {
    name: String,
    dict_path: PathBuf,
    weights_path: Option<PathBuf>,
}

#[cfg(aiocr_has_generated_rec)]
fn resolve_base_model(config: &TrainingConfig) -> Result<BaseModelSpec, TrainError> {
    if let Some(dir) = &config.base_model_dir {
        let model = AiModelInfo::load_from_dir(dir)?;
        return Ok(BaseModelSpec {
            name: model.name,
            dict_path: model.dict_path,
            weights_path: Some(model.weights_path),
        });
    }

    Ok(BaseModelSpec {
        name: "默认 PP-OCRv5 Server".to_string(),
        dict_path: PathBuf::from("models/ppocr_keys_v1.txt"),
        weights_path: None,
    })
}

#[cfg(aiocr_has_generated_rec)]
fn load_train_model<B: burn::tensor::backend::Backend>(
    device: &B::Device,
    weights_path: Option<&Path>,
) -> Result<aiocr_core::models::rec_generated::Model<B>, TrainError> {
    use burn::tensor::Bytes;

    if let Some(path) = weights_path {
        let bytes = std::fs::read(path).map_err(|err| {
            TrainError::Model(format!("读取基础模型权重 {} 失败: {err}", path.display()))
        })?;
        Ok(aiocr_core::models::rec_generated::Model::<B>::from_bytes(
            Bytes::from_bytes_vec(bytes),
            device,
        ))
    } else {
        Ok(aiocr_core::models::rec_generated::Model::<B>::from_embedded(device))
    }
}

#[cfg(aiocr_has_generated_rec)]
fn normalize_prob_dims<B: burn::tensor::backend::Backend>(
    probabilities: burn::prelude::Tensor<B, 3>,
) -> Result<burn::prelude::Tensor<B, 3>, TrainError> {
    let dims = probabilities.dims();
    if dims[0] != 1 {
        return Err(TrainError::Training(format!(
            "识别模型输出 batch 维异常: {:?}",
            dims
        )));
    }

    if dims[2] >= dims[1] {
        Ok(probabilities)
    } else {
        Ok(probabilities.swap_dims(1, 2))
    }
}

#[cfg(aiocr_has_generated_rec)]
fn build_aligned_targets(target: &[usize], time_steps: usize) -> Vec<i64> {
    if time_steps == 0 {
        return Vec::new();
    }

    if target.is_empty() {
        return vec![0; time_steps];
    }

    let mut aligned = vec![0i64; time_steps];
    for (idx, class_idx) in target.iter().enumerate() {
        let time_pos =
            (((idx as f32 + 0.5) / target.len() as f32) * time_steps as f32).floor() as usize;
        let time_pos = time_pos.min(time_steps - 1);
        aligned[time_pos] = *class_idx as i64;
    }
    aligned
}

#[cfg(aiocr_has_generated_rec)]
fn decode_output<B: burn::tensor::backend::Backend>(
    decoder: &CtcDecoder,
    probabilities: burn::prelude::Tensor<B, 3>,
) -> Result<(String, f32), TrainError> {
    let probabilities = normalize_prob_dims(probabilities)?;
    let dims = probabilities.dims();
    let data = probabilities
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| TrainError::Training(format!("导出识别输出失败: {err}")))?;
    Ok(decoder.decode_probabilities(&data, dims[2]))
}

#[cfg(aiocr_has_generated_rec)]
fn evaluate_accuracy<B: burn::tensor::backend::Backend, M>(
    model: &M,
    decoder: &CtcDecoder,
    items: &[OcrItem],
) -> Result<f32, TrainError>
where
    M: RecognizerForward<B>,
{
    if items.is_empty() {
        return Ok(1.0);
    }

    let device = Default::default();
    let mut correct = 0usize;

    for item in items {
        let image = load_image(&item.image_path)?;
        let input_data = preprocess_for_recognition(&image);
        let width = input_data.len() / (3 * 48);
        let input = burn::prelude::Tensor::<B, 4>::from_data(
            burn::prelude::TensorData::new(input_data, [1, 3, 48, width]),
            &device,
        );
        let probabilities = model.forward_recognizer(input);
        let (prediction, _confidence) = decode_output(decoder, probabilities)?;
        if prediction == item.label_text {
            correct += 1;
        }
    }

    Ok(correct as f32 / items.len() as f32)
}

#[cfg(aiocr_has_generated_rec)]
fn save_model_bytes<B: burn::tensor::backend::Backend, M>(model: &M) -> Result<Vec<u8>, TrainError>
where
    M: burn_store::ModuleSnapshot<B>,
{
    let mut store = burn_store::BurnpackStore::from_bytes(None);
    model
        .save_into(&mut store)
        .map_err(|err| TrainError::Export(format!("导出 AI 模型失败: {err}")))?;
    let bytes = store
        .get_bytes()
        .map_err(|err| TrainError::Export(format!("读取导出权重失败: {err}")))?;
    Ok(bytes.to_vec())
}

#[cfg(aiocr_has_generated_rec)]
fn build_model_name(config: &TrainingConfig, base_name: &str) -> String {
    let dataset_name = config
        .dataset_path
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| "dataset".to_string());
    format!("{base_name}-{dataset_name}-finetuned")
}

#[cfg(aiocr_has_generated_rec)]
fn split_dataset(items: &[OcrItem]) -> (Vec<OcrItem>, Vec<OcrItem>) {
    if items.len() <= 1 {
        return (items.to_vec(), items.to_vec());
    }

    let mut train = Vec::new();
    let mut validation = Vec::new();

    for (index, item) in items.iter().cloned().enumerate() {
        if index % 5 == 0 {
            validation.push(item);
        } else {
            train.push(item);
        }
    }

    if train.is_empty() {
        train = validation.clone();
    }
    if validation.is_empty() {
        validation = train.clone();
    }

    (train, validation)
}

#[cfg(aiocr_has_generated_rec)]
fn load_image(path: &PathBuf) -> Result<DynamicImage, TrainError> {
    image::open(path)
        .map_err(|err| TrainError::Dataset(format!("加载训练图片 {:?} 失败: {err}", path)))
}

#[cfg(aiocr_has_generated_rec)]
fn check_cancel(cancel_flag: Option<&AtomicBool>) -> Result<(), TrainError> {
    if cancel_flag.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        return Err(TrainError::Training("训练已取消".to_string()));
    }
    Ok(())
}

#[cfg(aiocr_has_generated_rec)]
fn decoder_ref(decoder: &CtcDecoder) -> &CtcDecoder {
    decoder
}

#[cfg(aiocr_has_generated_rec)]
trait RecognizerForward<B: burn::tensor::backend::Backend> {
    fn forward_recognizer(&self, input: burn::prelude::Tensor<B, 4>)
    -> burn::prelude::Tensor<B, 3>;
}

#[cfg(aiocr_has_generated_rec)]
impl<B: burn::tensor::backend::Backend> RecognizerForward<B>
    for aiocr_core::models::rec_generated::Model<B>
{
    fn forward_recognizer(
        &self,
        input: burn::prelude::Tensor<B, 4>,
    ) -> burn::prelude::Tensor<B, 3> {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(aiocr_has_generated_rec)]
    #[test]
    fn test_default_training_base_model_is_loadable() {
        use burn::backend::NdArray;

        let config = TrainingConfig::default();
        let base_model = resolve_base_model(&config).unwrap();
        assert_eq!(base_model.name, "默认 PP-OCRv5 Server");
        assert_eq!(
            base_model.dict_path,
            PathBuf::from("models/ppocr_keys_v1.txt")
        );

        let device = Default::default();
        let _model = load_train_model::<NdArray<f32>>(&device, None).unwrap();
    }
}
