#[cfg(aiocr_has_rec)]
use burn::tensor::Bytes;
use image::DynamicImage;
use std::path::PathBuf;
#[cfg(aiocr_has_rec)]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::Recognizer;
use crate::decode::CtcDecoder;
use crate::error::OcrError;
#[cfg(aiocr_has_rec)]
use crate::preprocess::preprocess_for_recognition;

#[cfg(aiocr_has_rec)]
struct BurnRecognizerRuntime {
    device: crate::models::BurnDevice,
    model: crate::models::rec_generated::Model<crate::models::BurnBackend>,
}

#[cfg(aiocr_has_rec)]
impl BurnRecognizerRuntime {
    fn load_embedded() -> Result<Self, OcrError> {
        let device = crate::models::default_device();
        let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::models::rec_generated::Model::<crate::models::BurnBackend>::from_embedded(
                &device,
            )
        }))
        .map_err(|_| OcrError::Inference("加载内嵌识别模型失败".to_string()))?;

        Ok(Self { device, model })
    }

    fn load_from_bytes(bytes: Bytes) -> Result<Self, OcrError> {
        let device = crate::models::default_device();
        let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::models::rec_generated::Model::<crate::models::BurnBackend>::from_bytes(
                bytes, &device,
            )
        }))
        .map_err(|_| OcrError::Inference("加载识别模型权重失败".to_string()))?;

        Ok(Self { device, model })
    }
}

#[derive(Debug, Clone)]
enum RecognizerWeights {
    Embedded,
    BurnpackFile(PathBuf),
}

/// 文本识别器（SVTR / CRNN）
///
/// 当 ONNX 模型可用时，使用 burn-onnx 生成的纯 Burn 模型。
pub struct TextRecognizer {
    decoder: CtcDecoder,
    weights: RecognizerWeights,
    model_name: String,
    #[cfg(aiocr_has_rec)]
    runtime: std::sync::Mutex<Option<BurnRecognizerRuntime>>,
    #[cfg(aiocr_has_rec)]
    logged_output_class_mismatch: AtomicBool,
}

impl TextRecognizer {
    pub fn new(decoder: CtcDecoder) -> Result<Self, OcrError> {
        Self::from_embedded(decoder)
    }

    pub fn from_embedded(decoder: CtcDecoder) -> Result<Self, OcrError> {
        Ok(Self {
            decoder,
            weights: RecognizerWeights::Embedded,
            model_name: "pp-ocrv5-rec".to_string(),
            #[cfg(aiocr_has_rec)]
            runtime: std::sync::Mutex::new(None),
            #[cfg(aiocr_has_rec)]
            logged_output_class_mismatch: AtomicBool::new(false),
        })
    }

    pub fn from_burnpack_file(
        decoder: CtcDecoder,
        weights_path: impl Into<PathBuf>,
        model_name: impl Into<String>,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            decoder,
            weights: RecognizerWeights::BurnpackFile(weights_path.into()),
            model_name: model_name.into(),
            #[cfg(aiocr_has_rec)]
            runtime: std::sync::Mutex::new(None),
            #[cfg(aiocr_has_rec)]
            logged_output_class_mismatch: AtomicBool::new(false),
        })
    }

    /// 当前仓库尚未集成 burn-onnx 生成代码时，识别模型不可用。
    pub fn is_model_available(&self) -> bool {
        if !Self::generated_model_available() {
            return false;
        }

        match &self.weights {
            RecognizerWeights::Embedded => true,
            RecognizerWeights::BurnpackFile(path) => path.exists(),
        }
    }

    pub fn generated_model_available() -> bool {
        crate::models::has_generated_recognizer()
    }

    /// 识别裁剪的文本行图片
    pub fn recognize(&self, crop: &DynamicImage) -> Result<(String, f32), OcrError> {
        #[cfg(aiocr_has_rec)]
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|err| OcrError::Inference(format!("锁定识别模型状态失败: {err}")))?;
            if runtime.is_none() {
                *runtime = Some(self.load_runtime()?);
            }
            let runtime = runtime.as_ref().expect("runtime just initialized");

            let input = preprocess_for_recognition(crop);
            let width = input.len() / (3 * 48);
            let tensor = crate::models::nchw_tensor(&input, [1, 3, 48, width], &runtime.device);
            let output = runtime.model.forward(tensor);
            let dims = output.dims();
            let probabilities = crate::models::tensor_to_vec(output)?;
            let dict_classes = self.decoder.num_classes();

            if dims[0] != 1 {
                return Err(OcrError::Inference(format!(
                    "识别模型输出 batch 维异常: expected 1, got {:?}",
                    dims
                )));
            }

            let (flattened, output_classes) = if dims[2] >= dims[1] {
                (probabilities, dims[2])
            } else {
                (
                    transpose_bct_to_btc(&probabilities, dims[1], dims[2]),
                    dims[1],
                )
            };

            if output_classes < dict_classes {
                return Err(OcrError::Inference(format!(
                    "识别模型输出类别数少于字典大小: output={:?}, dict_classes={dict_classes}",
                    dims,
                )));
            }
            if output_classes != dict_classes {
                let extra_classes = output_classes - dict_classes;
                if !self
                    .logged_output_class_mismatch
                    .swap(true, Ordering::Relaxed)
                {
                    if extra_classes == 1 {
                        tracing::info!(
                            "识别模型输出包含 1 个额外保留类，按兼容模式解码: output_classes={}, dict_classes={dict_classes}",
                            output_classes,
                        );
                    } else {
                        tracing::warn!(
                            "识别模型输出类别数与字典不完全一致: output_classes={}, dict_classes={dict_classes}",
                            output_classes,
                        );
                    }
                }
            }

            Ok(self
                .decoder
                .decode_probabilities(&flattened, output_classes))
        }

        #[cfg(not(aiocr_has_rec))]
        {
            Err(OcrError::ModelNotFound(format!(
                "识别模型未就绪，请先下载并构建 PaddleOCR ONNX 模型，或在 GUI 中训练并切换到本地 AI 模型后再识别。裁剪尺寸={}x{}",
                crop.width(),
                crop.height()
            )))
        }
    }

    /// 获取字典大小
    pub fn num_classes(&self) -> usize {
        self.decoder.num_classes()
    }

    #[cfg(aiocr_has_rec)]
    fn load_runtime(&self) -> Result<BurnRecognizerRuntime, OcrError> {
        match &self.weights {
            RecognizerWeights::Embedded => BurnRecognizerRuntime::load_embedded(),
            RecognizerWeights::BurnpackFile(path) => {
                let bytes = std::fs::read(path).map_err(|err| {
                    OcrError::ModelNotFound(format!("读取识别权重 {} 失败: {err}", path.display()))
                })?;
                BurnRecognizerRuntime::load_from_bytes(Bytes::from_bytes_vec(bytes))
            }
        }
    }
}

impl Recognizer for TextRecognizer {
    fn recognize(&self, crop: &DynamicImage) -> Result<(String, f32), OcrError> {
        TextRecognizer::recognize(self, crop)
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(aiocr_has_rec)]
fn transpose_bct_to_btc(values: &[f32], classes: usize, time_steps: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; values.len()];
    for class_idx in 0..classes {
        for step in 0..time_steps {
            transposed[step * classes + class_idx] = values[class_idx * time_steps + step];
        }
    }
    transposed
}

#[cfg(test)]
mod tests {
    #[cfg(aiocr_has_rec)]
    use super::*;

    #[cfg(aiocr_has_rec)]
    #[test]
    fn test_generated_recognizer_smoke() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                    160,
                    48,
                    image::Rgb([255, 255, 255]),
                ));
                let dict_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../models/ppocr_keys_v1.txt");
                let recognizer =
                    TextRecognizer::new(CtcDecoder::from_dict_file(&dict_path).unwrap()).unwrap();
                let (_text, confidence) = recognizer.recognize(&image).unwrap();
                assert!(confidence.is_finite());
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
