use image::DynamicImage;

use crate::config::DetectionResize;
use crate::error::OcrError;
use crate::types::{BoundingBox, ImageMeta};

const DETECTION_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DETECTION_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Burn 内嵌检测器编译期固定 512×512，方形填充路径使用此尺寸。
/// ONNX 检测器走 limit_side 动态尺寸（保宽高比），分辨率由质量预设的 det_long_side 决定。
pub(crate) const DETECTION_BURN_SQUARE_SIZE: u32 = 512;
// DBNet 下采样要求输入宽高为 32 的倍数。
const DETECTION_SIZE_MULTIPLE: u32 = 32;
const RECOGNITION_MIN_WIDTH: u32 = 48;
// 长文本行宽度上限：过窄会横向压缩导致 CER 上升。提高到 2048 缓解长行截断。
const RECOGNITION_MAX_WIDTH: u32 = 2048;
const RECOGNITION_WIDTH_BUCKET: u32 = 32;
const TEXT_REGION_HORIZONTAL_PAD_RATIO: f32 = 0.02;
const TEXT_REGION_VERTICAL_PAD_RATIO: f32 = 0.45;
const TEXT_REGION_MIN_PAD: f32 = 2.0;
const TEXT_REGION_MAX_HORIZONTAL_PAD: f32 = 12.0;
const TEXT_REGION_MIN_VERTICAL_PAD_CAP: f32 = 8.0;

/// 将图片预处理为检测模型输入
///
/// - `square_pad` 模式：等比缩放后居中填充成方形（Burn 内嵌检测器固定 512×512）。
/// - `limit_side` 模式：等比缩放使最长边 = 目标值，宽高各自向上对齐到 32 倍数，
///   不做方形填充，直接喂非方形 [1,3,H,W]（ONNX 动态尺寸）。保宽高比 + 高分辨率
///   能显著提升宽文档/小字的检测召回。
/// - 使用 ImageNet mean/std 归一化，形状 [1, 3, H, W] (NCHW, float32)。
pub fn preprocess_for_detection(
    img: &DynamicImage,
    resize: DetectionResize,
) -> (Vec<f32>, ImageMeta) {
    if resize.square_pad {
        preprocess_for_detection_square(img, resize.target_long_side.max(DETECTION_SIZE_MULTIPLE))
    } else {
        preprocess_for_detection_limit_side(img, resize.target_long_side, resize.allow_upscale)
    }
}

/// 方形居中填充（Burn 路径，固定尺寸）。
fn preprocess_for_detection_square(img: &DynamicImage, size: u32) -> (Vec<f32>, ImageMeta) {
    let (orig_w, orig_h) = (img.width(), img.height());

    let scale =
        (size as f32 / orig_w.max(1) as f32).min(size as f32 / orig_h.max(1) as f32);
    let content_w = ((orig_w as f32 * scale).round() as u32).clamp(1, size);
    let content_h = ((orig_h as f32 * scale).round() as u32).clamp(1, size);
    let pad_x = (size - content_w) / 2;
    let pad_y = (size - content_h) / 2;

    let resized = resize_to_rgb(img, content_w, content_h);
    let mut canvas = image::RgbImage::from_pixel(size, size, image::Rgb([255, 255, 255]));
    image::imageops::replace(&mut canvas, &resized, pad_x.into(), pad_y.into());

    let meta = ImageMeta {
        orig_width: orig_w,
        orig_height: orig_h,
        resized_width: size,
        resized_height: size,
        content_width: content_w,
        content_height: content_h,
        pad_x,
        pad_y,
        scale_x: orig_w as f32 / content_w as f32,
        scale_y: orig_h as f32 / content_h as f32,
    };

    (
        normalize_rgb_to_nchw_with_stats(&canvas, &DETECTION_MEAN, &DETECTION_STD),
        meta,
    )
}

/// limit_side 保宽高比缩放（ONNX 路径，动态尺寸）。
fn preprocess_for_detection_limit_side(
    img: &DynamicImage,
    target_long_side: u32,
    allow_upscale: bool,
) -> (Vec<f32>, ImageMeta) {
    let (orig_w, orig_h) = (img.width(), img.height());
    let long_side = orig_w.max(orig_h).max(1);

    // limit_type=max：默认仅缩小；allow_upscale 时也放大到目标最长边。
    let ratio = if allow_upscale {
        target_long_side as f32 / long_side as f32
    } else {
        (target_long_side as f32 / long_side as f32).min(1.0)
    };

    let content_w = ((orig_w as f32 * ratio).round() as u32).max(1);
    let content_h = ((orig_h as f32 * ratio).round() as u32).max(1);

    // 向上对齐到 32 倍数，多出的右/下区域作为白边填充。
    // 张量尺寸量化到 32 的倍数 + 最长边封顶，使 (h,w) 组合收敛到有限集合，
    // 避免 tract 按 (h,w) 缓存的推理计划无限膨胀。
    let tensor_w = round_up_to_multiple(content_w, DETECTION_SIZE_MULTIPLE);
    let tensor_h = round_up_to_multiple(content_h, DETECTION_SIZE_MULTIPLE);

    let resized = resize_to_rgb(img, content_w, content_h);
    let mut canvas = image::RgbImage::from_pixel(tensor_w, tensor_h, image::Rgb([255, 255, 255]));
    image::imageops::replace(&mut canvas, &resized, 0, 0);

    let meta = ImageMeta {
        orig_width: orig_w,
        orig_height: orig_h,
        resized_width: tensor_w,
        resized_height: tensor_h,
        // content 为对齐前的有效内容尺寸：scale 必须据此计算，否则 32 对齐的白边
        // 会让坐标整体缩放偏移。
        content_width: content_w,
        content_height: content_h,
        pad_x: 0,
        pad_y: 0,
        scale_x: orig_w as f32 / content_w as f32,
        scale_y: orig_h as f32 / content_h as f32,
    };

    (
        normalize_rgb_to_nchw_with_stats(&canvas, &DETECTION_MEAN, &DETECTION_STD),
        meta,
    )
}

fn round_up_to_multiple(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value.max(1);
    }
    value.max(1).div_ceil(multiple).saturating_mul(multiple)
}

/// 将裁剪的文本行图片预处理为分类模型输入
///
/// 输入尺寸: [1, 3, 48, 192]
pub fn preprocess_for_classification(crop: &DynamicImage) -> Vec<f32> {
    let resized = resize_to_rgb(crop, 192, 48);
    normalize_rgb_to_nchw(&resized)
}

/// 将裁剪的文本行图片预处理为识别模型输入
///
/// 输入尺寸: [1, 3, 48, W]，W 根据宽高比动态计算
pub fn preprocess_for_recognition(crop: &DynamicImage) -> Vec<f32> {
    let target_h = 48u32;
    let (content_w, target_w) = recognition_target_widths(crop, target_h);

    let resized = resize_to_rgb(crop, content_w, target_h);
    if content_w == target_w {
        return normalize_rgb_to_nchw(&resized);
    }

    // 分桶宽度大于内容宽度时，右侧用白色填充，保持文本宽高比不被横向拉伸。
    let mut canvas = image::RgbImage::from_pixel(target_w, target_h, image::Rgb([255, 255, 255]));
    image::imageops::replace(&mut canvas, &resized, 0, 0);
    normalize_rgb_to_nchw(&canvas)
}

/// 返回 (按宽高比缩放后的内容宽度, 分桶对齐后的张量宽度)。
fn recognition_target_widths(crop: &DynamicImage, target_h: u32) -> (u32, u32) {
    let ratio = crop.width() as f32 / crop.height().max(1) as f32;
    let content_w = ((target_h as f32 * ratio).round() as u32)
        .clamp(RECOGNITION_MIN_WIDTH, RECOGNITION_MAX_WIDTH);
    (content_w, bucket_width(content_w))
}

fn bucket_width(width: u32) -> u32 {
    if width <= RECOGNITION_MIN_WIDTH || width >= RECOGNITION_MAX_WIDTH {
        return width;
    }

    width
        .div_ceil(RECOGNITION_WIDTH_BUCKET)
        .saturating_mul(RECOGNITION_WIDTH_BUCKET)
        .min(RECOGNITION_MAX_WIDTH)
}

pub(crate) fn denormalize_detection_channel(channel: usize, value: f32) -> f32 {
    (value * DETECTION_STD[channel] + DETECTION_MEAN[channel]).clamp(0.0, 1.0)
}

/// 从检测框裁剪图片区域
///
/// 轴对齐框走快速 AABB 裁剪；旋转框（高精度档的最小面积矩形）走透视/旋转裁剪，
/// 把倾斜文本摆正后再送识别，显著降低背景噪声、提升识别率。
pub fn crop_text_region(img: &DynamicImage, bbox: &BoundingBox) -> Result<DynamicImage, OcrError> {
    if !bbox.is_axis_aligned() {
        return crop_rotated_text_region(img, bbox);
    }

    let rect = bbox
        .clamp(
            img.width().saturating_sub(1) as f32,
            img.height().saturating_sub(1) as f32,
        )
        .to_rect();

    let rect_w = (rect[2] - rect[0]).max(1.0);
    let rect_h = (rect[3] - rect[1]).max(1.0);
    let pad_x = (rect_w * TEXT_REGION_HORIZONTAL_PAD_RATIO)
        .clamp(TEXT_REGION_MIN_PAD, TEXT_REGION_MAX_HORIZONTAL_PAD);
    // 垂直 padding 上限按行高动态放宽，避免大字号行裁掉上下笔画。
    let max_vertical_pad = (rect_h * 0.10).max(TEXT_REGION_MIN_VERTICAL_PAD_CAP);
    let pad_y =
        (rect_h * TEXT_REGION_VERTICAL_PAD_RATIO).clamp(TEXT_REGION_MIN_PAD, max_vertical_pad);

    let x = (rect[0] - pad_x).floor().max(0.0) as u32;
    let y = (rect[1] - pad_y).floor().max(0.0) as u32;
    let x_max = (rect[2] + pad_x).ceil().max(rect[0] + 1.0) as u32;
    let y_max = (rect[3] + pad_y).ceil().max(rect[1] + 1.0) as u32;
    let w = x_max.saturating_sub(x).min(img.width().saturating_sub(x));
    let h = y_max.saturating_sub(y).min(img.height().saturating_sub(y));

    if w == 0 || h == 0 {
        return Err(OcrError::Preprocess("裁剪区域为空".to_string()));
    }

    Ok(img.crop_imm(x, y, w, h))
}

/// 旋转裁剪：把旋转四点框内的区域仿射采样为正立矩形（get_rotate_crop_image 等价）。
///
/// 先裁出框的轴对齐外接小区域（避免整图转换 RGB），再在小区域内做反向仿射 + 双线性采样。
fn crop_rotated_text_region(
    img: &DynamicImage,
    bbox: &BoundingBox,
) -> Result<DynamicImage, OcrError> {
    let clamped = bbox.clamp(
        img.width().saturating_sub(1) as f32,
        img.height().saturating_sub(1) as f32,
    );
    let rect = clamped.to_rect();

    // 加一点边距，避免采样到边界外。
    let margin = 2.0;
    let ax = (rect[0] - margin).floor().max(0.0) as u32;
    let ay = (rect[1] - margin).floor().max(0.0) as u32;
    let aw = ((rect[2] + margin).ceil() as u32)
        .saturating_sub(ax)
        .min(img.width().saturating_sub(ax))
        .max(1);
    let ah = ((rect[3] + margin).ceil() as u32)
        .saturating_sub(ay)
        .min(img.height().saturating_sub(ay))
        .max(1);

    let region = img.crop_imm(ax, ay, aw, ah).to_rgb8();

    // 框四点转换到小区域坐标系。
    let p = clamped.points;
    let local = |idx: usize| [p[idx][0] - ax as f32, p[idx][1] - ay as f32];
    let (p0, p1, p3) = (local(0), local(1), local(3));

    let dist = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let out_w = dist(p0, p1).round().max(1.0) as u32;
    let out_h = dist(p0, p3).round().max(1.0) as u32;

    let edge_u = [p1[0] - p0[0], p1[1] - p0[1]];
    let edge_v = [p3[0] - p0[0], p3[1] - p0[1]];

    let mut out = image::RgbImage::new(out_w, out_h);
    for oy in 0..out_h {
        let fy = (oy as f32 + 0.5) / out_h as f32;
        for ox in 0..out_w {
            let fx = (ox as f32 + 0.5) / out_w as f32;
            let sx = p0[0] + fx * edge_u[0] + fy * edge_v[0];
            let sy = p0[1] + fx * edge_u[1] + fy * edge_v[1];
            out.put_pixel(ox, oy, bilinear_sample_rgb(&region, sx, sy));
        }
    }

    Ok(DynamicImage::ImageRgb8(out))
}

/// 在 RGB 图上做双线性采样，坐标越界时夹取到边界。
fn bilinear_sample_rgb(img: &image::RgbImage, x: f32, y: f32) -> image::Rgb<u8> {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return image::Rgb([255, 255, 255]);
    }
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    let mut out = [0u8; 3];
    for (channel, value) in out.iter_mut().enumerate() {
        let p00 = img.get_pixel(x0, y0)[channel] as f32;
        let p10 = img.get_pixel(x1, y0)[channel] as f32;
        let p01 = img.get_pixel(x0, y1)[channel] as f32;
        let p11 = img.get_pixel(x1, y1)[channel] as f32;
        let top = p00 * (1.0 - wx) + p10 * wx;
        let bottom = p01 * (1.0 - wx) + p11 * wx;
        *value = (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
    }
    image::Rgb(out)
}

/// 将图像按比例缩放并填充到固定大小的灰度画布上。
///
/// 该函数用于模板特征提取，保持宽高比并尽量减少拉伸失真。
pub fn fit_to_canvas_grayscale(
    img: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> image::GrayImage {
    let gray = img.to_luma8();
    let (src_w, src_h) = (gray.width().max(1), gray.height().max(1));

    let scale = (target_width as f32 / src_w as f32)
        .min(target_height as f32 / src_h as f32)
        .max(f32::EPSILON);
    let resized_w = ((src_w as f32 * scale).round() as u32).clamp(1, target_width);
    let resized_h = ((src_h as f32 * scale).round() as u32).clamp(1, target_height);

    let resized = image::imageops::resize(
        &gray,
        resized_w,
        resized_h,
        image::imageops::FilterType::Triangle,
    );

    let background = background_luma(&gray);
    let mut canvas =
        image::GrayImage::from_pixel(target_width, target_height, image::Luma([background]));

    let offset_x = (target_width.saturating_sub(resized_w)) / 2;
    let offset_y = (target_height.saturating_sub(resized_h)) / 2;

    for y in 0..resized_h {
        for x in 0..resized_w {
            let pixel = *resized.get_pixel(x, y);
            canvas.put_pixel(offset_x + x, offset_y + y, pixel);
        }
    }

    canvas
}

/// 将灰度图转换为归一化的 ink 特征，值越大代表越像前景文本。
pub fn grayscale_to_ink(gray: &image::GrayImage) -> Vec<f32> {
    let mean = gray.as_raw().iter().map(|&v| v as f32 / 255.0).sum::<f32>()
        / gray.as_raw().len().max(1) as f32;

    let invert = mean > 0.5;
    gray.as_raw()
        .iter()
        .map(|&value| {
            let value = value as f32 / 255.0;
            let ink = if invert { 1.0 - value } else { value };
            ink.clamp(0.0, 1.0)
        })
        .collect()
}

/// 归一化图片到 NCHW 格式 [-1, 1]
fn normalize_rgb_to_nchw(img: &image::RgbImage) -> Vec<f32> {
    normalize_rgb_to_nchw_with_stats(img, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])
}

fn normalize_rgb_to_nchw_with_stats(
    img: &image::RgbImage,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Vec<f32> {
    let size = img.width() as usize * img.height() as usize;
    let mut data = vec![0.0f32; 3 * size];

    for (index, pixel) in img.as_raw().chunks_exact(3).enumerate() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        data[index] = (r - mean[0]) / std[0];
        data[size + index] = (g - mean[1]) / std[1];
        data[size * 2 + index] = (b - mean[2]) / std[2];
    }

    data
}

fn resize_to_rgb(img: &DynamicImage, width: u32, height: u32) -> image::RgbImage {
    match img {
        DynamicImage::ImageRgb8(rgb) => {
            image::imageops::resize(rgb, width, height, image::imageops::FilterType::Lanczos3)
        }
        _ => {
            let rgb = img.to_rgb8();
            image::imageops::resize(&rgb, width, height, image::imageops::FilterType::Lanczos3)
        }
    }
}

fn background_luma(gray: &image::GrayImage) -> u8 {
    let sum = gray.as_raw().iter().map(|&v| v as u64).sum::<u64>();
    (sum / gray.as_raw().len().max(1) as u64) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_to_canvas_preserves_target_size() {
        let img =
            DynamicImage::ImageLuma8(image::GrayImage::from_pixel(20, 10, image::Luma([255])));
        let canvas = fit_to_canvas_grayscale(&img, 64, 32);
        assert_eq!(canvas.dimensions(), (64, 32));
    }

    #[test]
    fn test_preprocess_for_detection_uses_imagenet_normalization() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            8,
            8,
            image::Rgb([255, 255, 255]),
        ));
        let (data, meta) = preprocess_for_detection(&img, DetectionResize::square(512));

        assert_eq!(meta.resized_width, DETECTION_BURN_SQUARE_SIZE);
        assert_eq!(meta.resized_height, DETECTION_BURN_SQUARE_SIZE);
        assert_eq!(meta.content_width, DETECTION_BURN_SQUARE_SIZE);
        assert_eq!(meta.content_height, DETECTION_BURN_SQUARE_SIZE);
        assert_eq!(meta.pad_x, 0);
        assert_eq!(meta.pad_y, 0);
        assert_eq!(
            data.len(),
            3 * DETECTION_BURN_SQUARE_SIZE as usize * DETECTION_BURN_SQUARE_SIZE as usize
        );

        let plane = DETECTION_BURN_SQUARE_SIZE as usize * DETECTION_BURN_SQUARE_SIZE as usize;
        let expected = [
            (1.0 - DETECTION_MEAN[0]) / DETECTION_STD[0],
            (1.0 - DETECTION_MEAN[1]) / DETECTION_STD[1],
            (1.0 - DETECTION_MEAN[2]) / DETECTION_STD[2],
        ];

        for channel in 0..3 {
            let actual = data[channel * plane];
            assert!((actual - expected[channel]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_preprocess_for_detection_keeps_aspect_ratio_with_padding() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            200,
            100,
            image::Rgb([255, 255, 255]),
        ));
        let (_data, meta) = preprocess_for_detection(&img, DetectionResize::square(512));

        assert_eq!(meta.resized_width, DETECTION_BURN_SQUARE_SIZE);
        assert_eq!(meta.resized_height, DETECTION_BURN_SQUARE_SIZE);
        assert_eq!(meta.content_width, DETECTION_BURN_SQUARE_SIZE);
        assert_eq!(meta.content_height, DETECTION_BURN_SQUARE_SIZE / 2);
        assert_eq!(meta.pad_x, 0);
        assert_eq!(meta.pad_y, DETECTION_BURN_SQUARE_SIZE / 4);
    }

    #[test]
    fn test_preprocess_for_detection_limit_side_preserves_aspect_and_aligns_to_32() {
        // 800x300 图，limit_side=640 不放大：ratio=640/800=0.8 → content 640x240，
        // 32 对齐后张量 640x256（高度 240→256 白边填充），pad=0，scale 用对齐前尺寸。
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            800,
            300,
            image::Rgb([255, 255, 255]),
        ));
        let (data, meta) = preprocess_for_detection(&img, DetectionResize::limit_side(640, false));

        assert_eq!(meta.content_width, 640);
        assert_eq!(meta.content_height, 240);
        assert_eq!(meta.resized_width, 640);
        assert_eq!(meta.resized_height, 256);
        assert_eq!(meta.pad_x, 0);
        assert_eq!(meta.pad_y, 0);
        assert!((meta.scale_x - 800.0 / 640.0).abs() < 1e-4);
        assert!((meta.scale_y - 300.0 / 240.0).abs() < 1e-4);
        assert_eq!(data.len(), 3 * 640 * 256);
    }

    #[test]
    fn test_preprocess_for_detection_limit_side_does_not_upscale_small_image() {
        // 100x60 小图，limit_side=1600 不放大：保持原尺寸，32 对齐到 128x64。
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            100,
            60,
            image::Rgb([255, 255, 255]),
        ));
        let (_data, meta) = preprocess_for_detection(&img, DetectionResize::limit_side(1600, false));

        assert_eq!(meta.content_width, 100);
        assert_eq!(meta.content_height, 60);
        assert_eq!(meta.resized_width, 128);
        assert_eq!(meta.resized_height, 64);
    }

    #[test]
    fn test_preprocess_for_classification_has_fixed_output_size() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            32,
            16,
            image::Rgb([255, 255, 255]),
        ));
        let data = preprocess_for_classification(&img);

        assert_eq!(data.len(), 3 * 48 * 192);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[48 * 192] - 1.0).abs() < 1e-6);
        assert!((data[48 * 192 * 2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_preprocess_for_recognition_clamps_target_width() {
        let narrow = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            10,
            100,
            image::Rgb([255, 255, 255]),
        ));
        let wide = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            4000,
            48,
            image::Rgb([255, 255, 255]),
        ));

        assert_eq!(
            preprocess_for_recognition(&narrow).len(),
            3 * 48 * RECOGNITION_MIN_WIDTH as usize
        );
        assert_eq!(
            preprocess_for_recognition(&wide).len(),
            3 * 48 * RECOGNITION_MAX_WIDTH as usize
        );
    }

    #[test]
    fn test_preprocess_for_recognition_buckets_target_width() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            97,
            48,
            image::Rgb([255, 255, 255]),
        ));

        assert_eq!(
            preprocess_for_recognition(&img).len(),
            3 * 48 * 128
        );
    }

    #[test]
    fn test_crop_text_region_adds_padding() {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            100,
            100,
            image::Rgb([255, 255, 255]),
        ));
        let bbox = BoundingBox::from_rect(40.0, 40.0, 10.0, 5.0);
        let crop = crop_text_region(&img, &bbox).unwrap();

        assert!(crop.width() > 10);
        assert!(crop.height() > 5);
    }
}
