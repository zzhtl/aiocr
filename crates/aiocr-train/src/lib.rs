//! AIOCR Train - 训练/微调模块
//!
//! 基于 Burn 框架实现本地 OCR 识别模型的训练和微调。

pub mod ai_model;
pub mod config;
pub mod ctc_loss;
pub mod data;
pub mod error;
pub mod model;
pub mod template;
pub mod training;

pub use ai_model::{AiModelInfo, AiModelManifest, AiModelMetrics, list_ai_models};
pub use config::TrainingConfig;
pub use template::{TemplateModelInfo, TemplateRecognizer, list_models};
pub use training::{LogCallback, Trainer, TrainingCallback, TrainingProgress, TrainingSummary};
