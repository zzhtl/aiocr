use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// OCR 引擎配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrConfig {
    /// 模型权重目录
    pub weights_dir: PathBuf,
    /// 字符字典路径
    pub dict_path: PathBuf,
    /// 检测阈值
    pub det_threshold: f32,
    /// 检测框阈值
    pub det_box_threshold: f32,
    /// 检测框扩展比例
    pub det_unclip_ratio: f32,
    /// 最大候选框数量
    pub det_max_candidates: usize,
    /// 分类置信度阈值
    pub cls_threshold: f32,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            weights_dir: PathBuf::from("models"),
            dict_path: PathBuf::from("models/ppocr_keys_v1.txt"),
            det_threshold: 0.3,
            det_box_threshold: 0.6,
            det_unclip_ratio: 1.5,
            det_max_candidates: 1000,
            cls_threshold: 0.9,
        }
    }
}

/// ONNX 模型文件路径配置
///
/// 用于直接指定外部 ONNX 文件路径，绕过编译时的 burn-onnx 转换，
/// 可以使用任意 PaddleOCR 兼容的 ONNX 模型文件。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct OnnxModelConfig {
    /// 检测模型路径（det.onnx）
    pub det_path: Option<PathBuf>,
    /// 识别模型路径（rec.onnx）
    pub rec_path: Option<PathBuf>,
    /// 方向分类模型路径（cls.onnx，可选）
    pub cls_path: Option<PathBuf>,
    /// 字符字典路径（dict.txt）
    pub dict_path: Option<PathBuf>,
}

impl OnnxModelConfig {
    /// 是否包含有效的检测和识别模型
    pub fn is_usable(&self) -> bool {
        self.det_path.as_ref().is_some_and(|p| p.exists())
            && self.rec_path.as_ref().is_some_and(|p| p.exists())
    }

    /// 快速从目录加载，自动寻找常见文件名
    pub fn from_dir(dir: &std::path::Path) -> Self {
        let try_path = |names: &[&str]| -> Option<PathBuf> {
            names.iter().map(|n| dir.join(n)).find(|p| p.exists())
        };

        Self {
            det_path: try_path(&["det.onnx"]),
            rec_path: try_path(&["rec.onnx"]),
            cls_path: try_path(&["cls.onnx"]),
            dict_path: try_path(&["ppocr_keys_v1.txt", "dict.txt"]),
        }
    }
}
