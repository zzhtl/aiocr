use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 识别质量预设：在精度与速度之间打包一组检测/后处理参数。
///
/// 纯 Rust CPU 推理较慢，检测分辨率越高越准但越慢。三档预设让用户一键取舍，
/// 默认 `High`（优先精度）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityPreset {
    /// 快速：较低检测分辨率，优先速度。
    Fast,
    /// 均衡：兼顾速度与精度。
    Balanced,
    /// 高精度：较高检测分辨率，优先识别效果（默认）。
    High,
}

impl Default for QualityPreset {
    fn default() -> Self {
        Self::High
    }
}

impl QualityPreset {
    /// 中文显示名。
    pub fn display_name(self) -> &'static str {
        match self {
            QualityPreset::Fast => "快速",
            QualityPreset::Balanced => "均衡",
            QualityPreset::High => "高精度",
        }
    }
}

/// 检测预处理的缩放策略。
///
/// - `square_pad = true`：等比缩放后居中填充成 `target_long_side` 方形（Burn 内嵌
///   检测器编译期固定 512×512，必须走这条）。
/// - `square_pad = false`：PaddleOCR `limit_type=max` 风格，等比缩放使最长边为
///   `target_long_side`，宽高各自向上对齐到 32 倍数，不做方形填充（ONNX 动态尺寸）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionResize {
    /// 目标最长边（或方形边长）。
    pub target_long_side: u32,
    /// 是否方形居中填充。
    pub square_pad: bool,
    /// 是否允许放大小于目标尺寸的图片（默认 false，仅缩小）。
    pub allow_upscale: bool,
}

impl DetectionResize {
    /// 方形居中填充（Burn 回退检测器专用，固定 512）。
    pub const fn square(size: u32) -> Self {
        Self {
            target_long_side: size,
            square_pad: true,
            allow_upscale: true,
        }
    }

    /// 保宽高比的 limit_side 缩放（ONNX 检测器）。
    pub const fn limit_side(target_long_side: u32, allow_upscale: bool) -> Self {
        Self {
            target_long_side,
            square_pad: false,
            allow_upscale,
        }
    }
}

/// 检测框形状模式。
///
/// - `AxisAligned`：轴对齐外接矩形（默认，快/均衡档，行为与历史一致）。
/// - `MinAreaRect`：最小面积旋转矩形（仅高精度档），配合透视裁剪提升倾斜文本识别率。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionBoxMode {
    AxisAligned,
    MinAreaRect,
}

/// 检测器构造参数（检测阈值 + 后处理 + 预处理缩放）。
#[derive(Debug, Clone, Copy)]
pub struct DetectionParams {
    /// 概率图二值化阈值。
    pub threshold: f32,
    /// 检测框平均分阈值。
    pub box_threshold: f32,
    /// 最大候选框数量。
    pub max_candidates: usize,
    /// Unclip 扩展比例。
    pub unclip_ratio: f32,
    /// 最小检测框面积（resized 尺度像素面积），小于此值丢弃。
    pub min_box_area: f32,
    /// 预处理缩放策略。
    pub resize: DetectionResize,
    /// 检测框形状模式。
    pub box_mode: DetectionBoxMode,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            box_threshold: 0.6,
            max_candidates: 1000,
            unclip_ratio: 1.5,
            min_box_area: 6.0,
            resize: DetectionResize::square(512),
            box_mode: DetectionBoxMode::AxisAligned,
        }
    }
}

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
    /// 检测预处理目标最长边（ONNX 路径，越大越准越慢）
    #[serde(default = "default_det_long_side")]
    pub det_long_side: u32,
    /// 是否允许放大小图（默认仅缩小）
    #[serde(default)]
    pub det_allow_upscale: bool,
    /// 最小检测框面积（resized 尺度像素面积）
    #[serde(default = "default_min_box_area")]
    pub min_box_area: f32,
    /// 识别置信度阈值，低于此值的文本不进入版式输出（0 = 不过滤）
    #[serde(default)]
    pub rec_score_threshold: f32,
    /// 是否使用最小面积旋转框 + 透视裁剪（仅高精度档，提升倾斜文本识别）
    #[serde(default)]
    pub det_rotated_boxes: bool,
}

fn default_det_long_side() -> u32 {
    1600
}

fn default_min_box_area() -> f32 {
    4.0
}

impl OcrConfig {
    /// 按质量预设构造配置。
    pub fn with_preset(preset: QualityPreset) -> Self {
        // (det_long_side, det_box_threshold, det_unclip_ratio, min_box_area, rotated_boxes)
        let (det_long_side, det_box_threshold, det_unclip_ratio, min_box_area, det_rotated_boxes) =
            match preset {
                QualityPreset::Fast => (960, 0.6, 1.5, 10.0, false),
                QualityPreset::Balanced => (1280, 0.6, 1.6, 6.0, false),
                QualityPreset::High => (1600, 0.5, 1.7, 4.0, true),
            };

        Self {
            weights_dir: PathBuf::from("models"),
            dict_path: PathBuf::from("models/ppocr_keys_v1.txt"),
            det_threshold: 0.3,
            det_box_threshold,
            det_unclip_ratio,
            det_max_candidates: 1000,
            cls_threshold: 0.9,
            det_long_side,
            det_allow_upscale: false,
            min_box_area,
            // 默认不过滤，避免误删低对比度的真实文本；可由调用方/GUI 按需开启。
            rec_score_threshold: 0.0,
            det_rotated_boxes,
        }
    }

    /// 构造检测器参数（ONNX 路径使用 limit_side 缩放）。
    pub fn detection_params(&self) -> DetectionParams {
        DetectionParams {
            threshold: self.det_threshold,
            box_threshold: self.det_box_threshold,
            max_candidates: self.det_max_candidates,
            unclip_ratio: self.det_unclip_ratio,
            min_box_area: self.min_box_area,
            resize: DetectionResize::limit_side(self.det_long_side, self.det_allow_upscale),
            box_mode: if self.det_rotated_boxes {
                DetectionBoxMode::MinAreaRect
            } else {
                DetectionBoxMode::AxisAligned
            },
        }
    }
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self::with_preset(QualityPreset::High)
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
