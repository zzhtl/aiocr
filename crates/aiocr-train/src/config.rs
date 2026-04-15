use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 训练配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// 数据集路径
    pub dataset_path: PathBuf,
    /// 模型输出目录
    pub artifact_dir: PathBuf,
    /// 当前用于继续训练的基础模型目录；为空时使用内嵌默认 AI 模型
    pub base_model_dir: Option<PathBuf>,
    /// 训练轮数
    pub num_epochs: usize,
    /// 批大小
    pub batch_size: usize,
    /// 学习率
    pub learning_rate: f64,
    /// 图片高度
    pub img_height: usize,
    /// 图片宽度
    pub img_width: usize,
    /// 随机种子
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::from("dataset"),
            artifact_dir: PathBuf::from("artifacts"),
            base_model_dir: None,
            num_epochs: 50,
            batch_size: 32,
            learning_rate: 1e-3,
            img_height: 32,
            img_width: 320,
            seed: 42,
        }
    }
}
