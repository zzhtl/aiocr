/// 训练错误类型
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("数据集错误: {0}")]
    Dataset(String),

    #[error("模型错误: {0}")]
    Model(String),

    #[error("训练错误: {0}")]
    Training(String),

    #[error("导出错误: {0}")]
    Export(String),

    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
}
