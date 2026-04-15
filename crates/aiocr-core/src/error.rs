/// OCR 错误类型
#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    #[error("模型未找到: {0}")]
    ModelNotFound(String),

    #[error("推理失败: {0}")]
    Inference(String),

    #[error("图片处理失败: {0}")]
    Image(#[from] image::ImageError),

    #[error("预处理失败: {0}")]
    Preprocess(String),

    #[error("后处理失败: {0}")]
    Postprocess(String),

    #[error("字典加载失败: {0}")]
    Dictionary(String),

    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
}
