use std::path::PathBuf;

use aiocr_core::types::OcrResult;
use aiocr_train::{AiModelInfo, TrainingConfig, TrainingProgress};

const DEFAULT_SERVER_DET_URL: &str =
    "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v5/det.onnx";
const DEFAULT_SERVER_REC_URL: &str =
    "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/rec.onnx";
const DEFAULT_SERVER_DICT_URL: &str =
    "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/dict.txt";

/// 应用状态。
pub struct AppState {
    /// 当前加载的图片路径
    pub image_path: Option<PathBuf>,
    /// 当前加载的图片数据
    pub image_data: Option<image::DynamicImage>,
    /// OCR 识别结果
    pub ocr_result: Option<OcrResult>,
    /// 当前任务状态
    pub task_status: TaskStatus,
    /// 状态消息
    pub status_message: String,
    /// 是否显示检测框
    pub show_bboxes: bool,
    /// 训练与模型管理状态
    pub training: TrainingState,
    /// 当前激活的引擎后端
    pub active_backend: EngineBackend,
    /// ONNX 外部模型配置
    pub onnx: OnnxState,
    /// 模型下载中心状态
    pub model_hub: ModelHubState,
}

/// 当前激活的 OCR 引擎后端
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum EngineBackend {
    /// 默认混合链路（ONNX det + Burn rec）
    #[default]
    BurnDefault,
    /// 本地训练的 AI 模型（Burn 微调）
    LocalAi,
    /// 外部 ONNX 模型文件（tract-onnx 推理）
    Onnx,
}

impl EngineBackend {
    pub fn display_name(&self) -> &str {
        match self {
            Self::BurnDefault => "默认 PP-OCRv5 Server（ONNX det + Burn rec）",
            Self::LocalAi => "本地训练 AI 模型",
            Self::Onnx => "外部 ONNX 模型",
        }
    }
}

/// ONNX 外部模型路径状态
#[derive(Debug, Clone, Default)]
pub struct OnnxState {
    /// 检测模型路径（det.onnx）
    pub det_path: Option<PathBuf>,
    /// 识别模型路径（rec.onnx）
    pub rec_path: Option<PathBuf>,
    /// 字符字典路径
    pub dict_path: Option<PathBuf>,
    /// 模型所在目录（用于批量加载）
    pub model_dir: Option<PathBuf>,
}

impl OnnxState {
    /// 是否已配置最低可用的文件
    pub fn is_usable(&self) -> bool {
        self.det_path.as_ref().is_some_and(|p| p.exists())
            && self.rec_path.as_ref().is_some_and(|p| p.exists())
    }

    /// 从目录批量设置路径
    pub fn load_from_dir(&mut self, dir: PathBuf) {
        let try_path = |name: &str| {
            let p = dir.join(name);
            p.exists().then_some(p)
        };
        self.det_path = try_path("det.onnx");
        self.rec_path = try_path("rec.onnx");
        self.dict_path = try_path("ppocr_keys_v1.txt").or_else(|| try_path("dict.txt"));
        self.model_dir = Some(dir);
    }

    /// 状态描述
    pub fn status(&self) -> String {
        if self.is_usable() {
            format!(
                "已配置: det={}, rec={}",
                self.det_path
                    .as_ref()
                    .map(|p| p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned())
                    .unwrap_or_default(),
                self.rec_path
                    .as_ref()
                    .map(|p| p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned())
                    .unwrap_or_default(),
            )
        } else {
            "未配置 ONNX 模型（需要 det.onnx + rec.onnx）".to_string()
        }
    }
}

/// 训练与模型管理状态。
#[derive(Default)]
pub struct TrainingState {
    /// 训练参数
    pub config: TrainingConfig,
    /// 当前训练进度
    pub progress: Option<TrainingProgress>,
    /// 已发现的本地 AI 模型
    pub available_models: Vec<AiModelInfo>,
    /// 当前列表中选中的模型目录
    pub selected_model_dir: Option<PathBuf>,
    /// 当前激活的识别模型目录
    pub active_model_dir: Option<PathBuf>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            image_path: None,
            image_data: None,
            ocr_result: None,
            task_status: TaskStatus::Idle,
            status_message: "就绪，可拖拽图片或按 Ctrl+V / Cmd+V 粘贴截图".to_string(),
            show_bboxes: true,
            training: TrainingState::default(),
            active_backend: EngineBackend::default(),
            onnx: OnnxState::default(),
            model_hub: ModelHubState::default(),
        }
    }
}

impl TrainingState {
    pub fn selected_model(&self) -> Option<&AiModelInfo> {
        self.selected_model_dir.as_ref().and_then(|selected| {
            self.available_models
                .iter()
                .find(|model| &model.model_dir == selected)
        })
    }

    pub fn active_model(&self) -> Option<&AiModelInfo> {
        self.active_model_dir.as_ref().and_then(|active| {
            self.available_models
                .iter()
                .find(|model| &model.model_dir == active)
        })
    }

    pub fn set_available_models(&mut self, models: Vec<AiModelInfo>) {
        self.available_models = models;

        if let Some(selected) = &self.selected_model_dir
            && !self
                .available_models
                .iter()
                .any(|model| &model.model_dir == selected)
        {
            self.selected_model_dir = None;
        }

        if let Some(active) = &self.active_model_dir
            && !self
                .available_models
                .iter()
                .any(|model| &model.model_dir == active)
        {
            self.active_model_dir = None;
        }

        if self.selected_model_dir.is_none()
            && let Some(first) = self.available_models.first()
        {
            self.selected_model_dir = Some(first.model_dir.clone());
        }
    }
}

/// 模型下载中心状态
pub struct ModelHubState {
    /// 检测模型 URL
    pub det_url: String,
    /// 识别模型 URL
    pub rec_url: String,
    /// 字典文件 URL（可选）
    pub dict_url: String,
    /// 模型保存目录
    pub save_dir: PathBuf,
    /// 当前下载状态
    pub download_status: DownloadStatus,
}

impl Default for ModelHubState {
    fn default() -> Self {
        Self {
            det_url: DEFAULT_SERVER_DET_URL.to_string(),
            rec_url: DEFAULT_SERVER_REC_URL.to_string(),
            dict_url: DEFAULT_SERVER_DICT_URL.to_string(),
            save_dir: PathBuf::from("models"),
            download_status: DownloadStatus::Idle,
        }
    }
}

/// 下载进度状态
pub enum DownloadStatus {
    Idle,
    Downloading {
        component: String,
        downloaded: u64,
        total: u64,
    },
    Complete,
    Error(String),
}

impl DownloadStatus {
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Downloading { .. })
    }
}

/// 任务状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Idle,
    Loading,
    Recognizing,
    Training,
    Error(String),
}

impl TaskStatus {
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Loading | Self::Recognizing | Self::Training)
    }
}
