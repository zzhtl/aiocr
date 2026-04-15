use serde::{Deserialize, Serialize};

/// 文本区域检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    /// 检测框
    pub bbox: BoundingBox,
    /// 检测置信度
    pub confidence: f32,
    /// 识别文本
    pub text: String,
    /// 文本方向
    pub direction: TextDirection,
}

/// 四点边界框（顺时针，从左上角开始）
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub points: [[f32; 2]; 4],
}

impl BoundingBox {
    /// 通过轴对齐矩形创建边界框。
    pub fn from_rect(x: f32, y: f32, width: f32, height: f32) -> Self {
        let x_max = x + width;
        let y_max = y + height;
        Self {
            points: [[x, y], [x_max, y], [x_max, y_max], [x, y_max]],
        }
    }

    /// 计算边界框面积
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// 获取边界框宽度。
    pub fn width(&self) -> f32 {
        ((self.points[1][0] - self.points[0][0]).powi(2)
            + (self.points[1][1] - self.points[0][1]).powi(2))
        .sqrt()
    }

    /// 获取边界框高度。
    pub fn height(&self) -> f32 {
        ((self.points[3][0] - self.points[0][0]).powi(2)
            + (self.points[3][1] - self.points[0][1]).powi(2))
        .sqrt()
    }

    /// 获取最小外接矩形 (x_min, y_min, x_max, y_max)
    pub fn to_rect(&self) -> [f32; 4] {
        let x_min = self.points.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        let y_min = self.points.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let x_max = self.points.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        let y_max = self.points.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        [x_min, y_min, x_max, y_max]
    }

    /// 将边界框裁剪到指定图片尺寸范围内。
    pub fn clamp(self, max_width: f32, max_height: f32) -> Self {
        let mut clamped = self;
        for point in &mut clamped.points {
            point[0] = point[0].clamp(0.0, max_width);
            point[1] = point[1].clamp(0.0, max_height);
        }
        clamped
    }

    /// 获取左上角 x。
    pub fn left(&self) -> f32 {
        self.to_rect()[0]
    }

    /// 获取左上角 y。
    pub fn top(&self) -> f32 {
        self.to_rect()[1]
    }

    /// 获取右下角 x。
    pub fn right(&self) -> f32 {
        self.to_rect()[2]
    }

    /// 获取右下角 y。
    pub fn bottom(&self) -> f32 {
        self.to_rect()[3]
    }

    /// 获取中心点 x。
    pub fn center_x(&self) -> f32 {
        let rect = self.to_rect();
        (rect[0] + rect[2]) * 0.5
    }

    /// 获取中心点 y。
    pub fn center_y(&self) -> f32 {
        let rect = self.to_rect();
        (rect[1] + rect[3]) * 0.5
    }

    /// 与另一个边界框求并集。
    pub fn union(&self, other: &Self) -> Self {
        let rect_a = self.to_rect();
        let rect_b = other.to_rect();
        Self::from_rect(
            rect_a[0].min(rect_b[0]),
            rect_a[1].min(rect_b[1]),
            rect_a[2].max(rect_b[2]) - rect_a[0].min(rect_b[0]),
            rect_a[3].max(rect_b[3]) - rect_a[1].min(rect_b[1]),
        )
    }
}

/// 文本方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextDirection {
    Horizontal,
    Vertical,
}

/// OCR 完整识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    /// 所有检测到的文本区域
    pub regions: Vec<TextRegion>,
    /// 拼接后的完整文本
    pub full_text: String,
    /// 处理耗时（毫秒）
    pub elapsed_ms: u64,
}

/// 图片预处理元数据（用于坐标映射）
#[derive(Debug, Clone)]
pub struct ImageMeta {
    /// 原始图片宽度
    pub orig_width: u32,
    /// 原始图片高度
    pub orig_height: u32,
    /// 模型输入宽度
    pub resized_width: u32,
    /// 模型输入高度
    pub resized_height: u32,
    /// 实际内容区域宽度
    pub content_width: u32,
    /// 实际内容区域高度
    pub content_height: u32,
    /// 左侧填充
    pub pad_x: u32,
    /// 顶部填充
    pub pad_y: u32,
    /// 水平缩放比（内容区域到原图）
    pub scale_x: f32,
    /// 垂直缩放比（内容区域到原图）
    pub scale_y: f32,
}
