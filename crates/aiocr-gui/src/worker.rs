use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use aiocr_core::config::OnnxModelConfig;
use aiocr_core::decode::CtcDecoder;
use aiocr_core::models::classifier::DirectionClassifier;
use aiocr_core::models::detector::TextDetector;
use aiocr_core::models::recognizer::TextRecognizer;
use aiocr_core::{OcrConfig, OcrEngine, Recognizer};
use aiocr_train::{
    AiModelInfo, Trainer, TrainingCallback, TrainingConfig, TrainingProgress, TrainingSummary,
    list_ai_models,
};

const OCR_THREAD_STACK_SIZE: usize = 64 * 1024 * 1024;

/// 后台任务消息。
pub enum WorkerMessage {
    /// 图片加载完成
    ImageLoaded(PathBuf, image::DynamicImage),
    /// OCR 识别完成
    OcrComplete(aiocr_core::types::OcrResult),
    /// 训练进度
    TrainingProgress(TrainingProgress),
    /// 训练完成
    TrainingComplete(TrainingSummary),
    /// 模型列表刷新
    ModelsLoaded(Vec<AiModelInfo>),
    /// 目录选择完成
    DirectorySelected(DirectoryKind, PathBuf),
    /// 模型下载进度
    DownloadProgress {
        component: String,
        downloaded: u64,
        total: u64,
    },
    /// 模型下载完成，返回保存目录
    DownloadComplete(PathBuf),
    /// 错误
    Error(String),
}

/// 目录选择类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryKind {
    Dataset,
    Artifact,
    OnnxModel,
    /// 模型下载中心保存目录
    ModelHubSave,
}

/// OCR 引擎选择规格。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EngineSpec {
    /// 默认混合链路（models/ 下的 ONNX det + Burn rec）
    Default,
    /// 本地训练的 AI 模型（Burn 微调）
    LocalAi(PathBuf),
    /// 外部 ONNX 模型文件
    Onnx {
        det_path: Option<PathBuf>,
        rec_path: Option<PathBuf>,
        dict_path: Option<PathBuf>,
    },
}

impl EngineSpec {
    fn display_name(&self) -> String {
        match self {
            Self::Default => "默认 PP-OCRv5 Server".to_string(),
            Self::LocalAi(path) => format!("AI 模型 {}", path.display()),
            Self::Onnx {
                det_path, rec_path, ..
            } => format!(
                "ONNX [det={}, rec={}]",
                det_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
                rec_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
            ),
        }
    }
}

/// 后台工作线程管理。
pub struct Worker {
    sender: mpsc::Sender<WorkerMessage>,
    training_cancel: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    ocr_engines: Arc<Mutex<HashMap<EngineSpec, Arc<OcrEngine>>>>,
}

impl Worker {
    pub fn new(sender: mpsc::Sender<WorkerMessage>) -> Self {
        Self {
            sender,
            training_cancel: Arc::new(Mutex::new(None)),
            ocr_engines: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 在后台线程打开文件对话框并加载图片。
    pub fn open_image(&self) {
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            let file = rfd::FileDialog::new()
                .add_filter("图片", &["png", "jpg", "jpeg", "webp", "bmp"])
                .pick_file();

            if let Some(path) = file {
                send_loaded_image(&sender, path);
            }
        });
    }

    /// 从指定路径异步加载图片。
    pub fn load_image_path(&self, path: PathBuf) {
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            send_loaded_image(&sender, path);
        });
    }

    /// 在后台线程选择目录。
    pub fn pick_directory(&self, kind: DirectoryKind, initial_dir: Option<PathBuf>) {
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            let mut dialog = rfd::FileDialog::new();
            if let Some(initial_dir) = initial_dir {
                dialog = dialog.set_directory(initial_dir);
            }

            if let Some(path) = dialog.pick_folder() {
                let _ = sender.send(WorkerMessage::DirectorySelected(kind, path));
            }
        });
    }

    /// 执行 OCR 识别。
    pub fn run_ocr(&self, image: image::DynamicImage, spec: EngineSpec) {
        let sender = self.sender.clone();
        let ocr_engines = self.ocr_engines.clone();
        let spawn_result = std::thread::Builder::new()
            .name("aiocr-ocr-worker".to_string())
            .stack_size(OCR_THREAD_STACK_SIZE)
            .spawn(move || match get_or_build_engine(&ocr_engines, spec) {
                Ok(engine) => match engine.run(&image) {
                    Ok(result) => {
                        let _ = sender.send(WorkerMessage::OcrComplete(result));
                    }
                    Err(err) => {
                        let _ = sender.send(WorkerMessage::Error(format!("OCR 识别失败: {err}")));
                    }
                },
                Err(err) => {
                    let _ = sender.send(WorkerMessage::Error(err));
                }
            });

        if let Err(err) = spawn_result {
            let _ = self.sender.send(WorkerMessage::Error(format!(
                "启动 OCR 后台线程失败: {err} (stack_size={}MB)",
                OCR_THREAD_STACK_SIZE / 1024 / 1024
            )));
        }
    }

    /// 启动训练任务。
    pub fn train(&self, config: TrainingConfig) {
        let sender = self.sender.clone();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel_holder = self.training_cancel.clone();
        if let Ok(mut slot) = cancel_holder.lock() {
            *slot = Some(cancel_flag.clone());
        }

        std::thread::spawn(move || {
            let mut callback = ChannelCallback {
                sender: sender.clone(),
            };
            let trainer = Trainer::new(config);
            let result = trainer.train_with_cancel(&mut callback, Some(cancel_flag.as_ref()));
            if let Err(err) = result {
                let _ = sender.send(WorkerMessage::Error(format!("训练失败: {err}")));
            }
            if let Ok(mut slot) = cancel_holder.lock() {
                *slot = None;
            }
        });
    }

    /// 取消当前训练任务。
    pub fn cancel_training(&self) {
        if let Ok(slot) = self.training_cancel.lock()
            && let Some(flag) = &*slot
        {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// 刷新本地模型列表。
    pub fn refresh_models(&self, artifact_dir: PathBuf) {
        let sender = self.sender.clone();
        std::thread::spawn(move || match list_ai_models(&artifact_dir) {
            Ok(models) => {
                let _ = sender.send(WorkerMessage::ModelsLoaded(models));
            }
            Err(err) => {
                let _ = sender.send(WorkerMessage::Error(format!("加载模型列表失败: {err}")));
            }
        });
    }

    /// 在后台线程下载 PaddleOCR ONNX 模型文件。
    pub fn download_models(
        &self,
        det_url: String,
        rec_url: String,
        dict_url: Option<String>,
        save_dir: PathBuf,
    ) {
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            if let Err(err) =
                do_download(&sender, &det_url, &rec_url, dict_url.as_deref(), &save_dir)
            {
                let _ = sender.send(WorkerMessage::Error(format!("模型下载失败: {err}")));
            }
        });
    }

    /// 清除指定 spec 的引擎缓存（模型文件变更后调用）。
    pub fn invalidate_onnx_cache(&self) {
        if let Ok(mut cache) = self.ocr_engines.lock() {
            cache.retain(|k, _| !matches!(k, EngineSpec::Onnx { .. }));
        }
    }
}

struct ChannelCallback {
    sender: mpsc::Sender<WorkerMessage>,
}

impl TrainingCallback for ChannelCallback {
    fn on_epoch_start(&mut self, _epoch: usize, _total_epochs: usize) {}

    fn on_batch_end(&mut self, progress: &TrainingProgress) {
        let _ = self
            .sender
            .send(WorkerMessage::TrainingProgress(progress.clone()));
    }

    fn on_epoch_end(&mut self, progress: &TrainingProgress) {
        let _ = self
            .sender
            .send(WorkerMessage::TrainingProgress(progress.clone()));
    }

    fn on_training_complete(&mut self, summary: &TrainingSummary) {
        let _ = self
            .sender
            .send(WorkerMessage::TrainingComplete(summary.clone()));
    }
}

fn build_engine(spec: &EngineSpec) -> Result<OcrEngine, String> {
    let config = OcrConfig::default();

    match spec {
        EngineSpec::Default => build_default_engine(&config),
        EngineSpec::LocalAi(model_dir) => build_local_ai_engine(&config, model_dir),
        EngineSpec::Onnx {
            det_path,
            rec_path,
            dict_path,
        } => build_onnx_engine(&config, det_path, rec_path, dict_path),
    }
}

fn build_default_engine(config: &OcrConfig) -> Result<OcrEngine, String> {
    OcrEngine::from_config(config).map_err(|err| format!("初始化默认 OCR 引擎失败: {err}"))
}

fn build_local_ai_engine(config: &OcrConfig, model_dir: &PathBuf) -> Result<OcrEngine, String> {
    let detector = Box::new(
        TextDetector::new(
            config.det_threshold,
            config.det_box_threshold,
            config.det_max_candidates,
            config.det_unclip_ratio,
        )
        .map_err(|err| format!("初始化检测器失败: {err}"))?,
    );
    let classifier = DirectionClassifier::new(config.cls_threshold)
        .map_err(|err| format!("初始化方向分类器失败: {err}"))?;
    let model = AiModelInfo::load_from_dir(model_dir)
        .map_err(|err| format!("加载 AI 模型 {:?} 失败: {err}", model_dir))?;
    let decoder = CtcDecoder::from_dict_or_builtin(&model.dict_path)
        .map_err(|err| format!("加载模型字典失败: {err}"))?;
    let recognizer: Box<dyn Recognizer> = Box::new(
        TextRecognizer::from_burnpack_file(decoder, model.weights_path, model.name)
            .map_err(|err| format!("初始化 AI 识别模型失败: {err}"))?,
    );

    Ok(OcrEngine::new(detector, classifier, recognizer))
}

fn build_onnx_engine(
    config: &OcrConfig,
    det_path: &Option<PathBuf>,
    rec_path: &Option<PathBuf>,
    dict_path: &Option<PathBuf>,
) -> Result<OcrEngine, String> {
    let onnx_config = OnnxModelConfig {
        det_path: det_path.clone(),
        rec_path: rec_path.clone(),
        cls_path: None,
        dict_path: dict_path.clone(),
    };

    OcrEngine::from_onnx(config, &onnx_config)
        .map_err(|err| format!("初始化 ONNX OCR 引擎失败: {err}"))
}

fn get_or_build_engine(
    cache: &Arc<Mutex<HashMap<EngineSpec, Arc<OcrEngine>>>>,
    spec: EngineSpec,
) -> Result<Arc<OcrEngine>, String> {
    if let Ok(cache) = cache.lock()
        && let Some(engine) = cache.get(&spec)
    {
        tracing::debug!("复用 OCR 引擎缓存: {}", spec.display_name());
        return Ok(engine.clone());
    }

    tracing::info!("初始化 OCR 引擎: {}", spec.display_name());
    let engine = Arc::new(build_engine(&spec)?);

    let mut cache = cache
        .lock()
        .map_err(|err| format!("锁定 OCR 引擎缓存失败: {err}"))?;

    if let Some(existing) = cache.get(&spec) {
        Ok(existing.clone())
    } else {
        cache.insert(spec, engine.clone());
        Ok(engine)
    }
}

fn do_download(
    sender: &mpsc::Sender<WorkerMessage>,
    det_url: &str,
    rec_url: &str,
    dict_url: Option<&str>,
    save_dir: &std::path::Path,
) -> Result<(), String> {
    std::fs::create_dir_all(save_dir).map_err(|e| format!("创建保存目录失败: {e}"))?;

    download_file(
        sender,
        det_url,
        &save_dir.join("det.onnx"),
        "检测模型 (det.onnx)",
    )?;
    download_file(
        sender,
        rec_url,
        &save_dir.join("rec.onnx"),
        "识别模型 (rec.onnx)",
    )?;

    if let Some(url) = dict_url {
        if !url.is_empty() {
            download_file(sender, url, &save_dir.join("ppocr_keys_v1.txt"), "字典文件")?;
        }
    }

    let _ = sender.send(WorkerMessage::DownloadComplete(save_dir.to_path_buf()));
    Ok(())
}

fn download_file(
    sender: &mpsc::Sender<WorkerMessage>,
    url: &str,
    save_path: &std::path::Path,
    component: &str,
) -> Result<(), String> {
    use std::io::{Read, Write};

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP 请求失败 ({component}): {e}"))?;

    let total: u64 = response
        .header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut reader = response.into_reader();
    let mut file =
        std::fs::File::create(save_path).map_err(|e| format!("创建文件失败 ({component}): {e}"))?;

    let mut downloaded = 0u64;
    let mut buf = vec![0u8; 65536];

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("读取数据失败: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("写入文件失败: {e}"))?;
        downloaded += n as u64;
        let _ = sender.send(WorkerMessage::DownloadProgress {
            component: component.to_string(),
            downloaded,
            total,
        });
    }

    Ok(())
}

fn send_loaded_image(sender: &mpsc::Sender<WorkerMessage>, path: PathBuf) {
    match image::open(&path) {
        Ok(img) => {
            let _ = sender.send(WorkerMessage::ImageLoaded(path, img));
        }
        Err(err) => {
            let _ = sender.send(WorkerMessage::Error(format!("加载图片失败: {err}")));
        }
    }
}
