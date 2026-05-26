// burn-onnx 生成的模型代码在编译时由 build.rs 生成。
// 当前仓库默认仍使用纯 Rust 后备实现；当 ONNX 文件存在时，这里会暴露生成模块，
// 后续只需把真实推理逻辑接到对应封装层即可。

use burn::backend::NdArray;
use burn::prelude::{Backend, Tensor, TensorData};

pub mod classifier;
pub mod detector;
pub mod onnx_backend;
pub mod recognizer;

pub use onnx_backend::{OnnxDetector, OnnxRecognizer};

#[cfg(aiocr_has_det)]
pub mod det_generated {
    include!(concat!(env!("OUT_DIR"), "/models/det/det.rs"));
}

#[cfg(aiocr_has_cls)]
pub mod cls_generated {
    include!(concat!(env!("OUT_DIR"), "/models/cls/cls.rs"));
}

#[cfg(aiocr_has_rec)]
pub mod rec_generated {
    include!(concat!(env!("OUT_DIR"), "/models/rec/rec.rs"));
}

pub const fn has_generated_detector() -> bool {
    cfg!(aiocr_has_det)
}

pub const fn has_generated_classifier() -> bool {
    cfg!(aiocr_has_cls)
}

pub const fn has_generated_recognizer() -> bool {
    cfg!(aiocr_has_rec)
}

pub(crate) type BurnBackend = NdArray<f32>;
pub(crate) type BurnDevice = <BurnBackend as Backend>::Device;

/// 批量推理时单次 forward 的最大样本数，限制激活内存峰值。
pub(crate) const MAX_INFERENCE_BATCH: usize = 32;

pub(crate) fn default_device() -> BurnDevice {
    Default::default()
}

pub(crate) fn nchw_tensor(
    data: &[f32],
    shape: [usize; 4],
    device: &BurnDevice,
) -> Tensor<BurnBackend, 4> {
    nchw_tensor_owned(data.to_vec(), shape, device)
}

pub(crate) fn nchw_tensor_owned(
    data: Vec<f32>,
    shape: [usize; 4],
    device: &BurnDevice,
) -> Tensor<BurnBackend, 4> {
    Tensor::from_data(TensorData::new(data, shape), device)
}

pub(crate) fn tensor_to_vec<const D: usize>(
    tensor: Tensor<BurnBackend, D>,
) -> Result<Vec<f32>, crate::error::OcrError> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| crate::error::OcrError::Inference(format!("导出模型输出张量失败: {err}")))
}
