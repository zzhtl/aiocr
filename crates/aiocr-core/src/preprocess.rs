use image::DynamicImage;

use crate::error::OcrError;
use crate::types::{BoundingBox, ImageMeta};

const DETECTION_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DETECTION_STD: [f32; 3] = [0.229, 0.224, 0.225];
// Server det 模型在 736 上效果最好，但纯 Rust 推理代价过高。
// 默认降到 512，保留 server 模型本身，不退回 mobile。
const DETECTION_TARGET_SIZE: u32 = 512;
const RECOGNITION_MIN_WIDTH: u32 = 48;
const RECOGNITION_MAX_WIDTH: u32 = 1280;
const TEXT_REGION_HORIZONTAL_PAD_RATIO: f32 = 0.02;
const TEXT_REGION_VERTICAL_PAD_RATIO: f32 = 0.45;
const TEXT_REGION_MIN_PAD: f32 = 2.0;
const TEXT_REGION_MAX_HORIZONTAL_PAD: f32 = 12.0;
const TEXT_REGION_MAX_VERTICAL_PAD: f32 = 8.0;

/// 将图片预处理为检测模型输入
///
/// PP-OCRv5 Server 检测模型要求：
/// - 固定输入到 512x512
/// - 保持宽高比后居中填充
/// - 使用 ImageNet mean/std 归一化
/// - 形状 [1, 3, H, W] (NCHW, float32)
pub fn preprocess_for_detection(img: &DynamicImage) -> (Vec<f32>, ImageMeta) {
    let (orig_w, orig_h) = (img.width(), img.height());

    let scale = (DETECTION_TARGET_SIZE as f32 / orig_w.max(1) as f32)
        .min(DETECTION_TARGET_SIZE as f32 / orig_h.max(1) as f32);
    let content_w = ((orig_w as f32 * scale).round() as u32).clamp(1, DETECTION_TARGET_SIZE);
    let content_h = ((orig_h as f32 * scale).round() as u32).clamp(1, DETECTION_TARGET_SIZE);
    let pad_x = (DETECTION_TARGET_SIZE - content_w) / 2;
    let pad_y = (DETECTION_TARGET_SIZE - content_h) / 2;

    let source = img.to_rgb8();
    let resized = image::imageops::resize(
        &source,
        content_w,
        content_h,
        image::imageops::FilterType::Lanczos3,
    );
    let mut canvas = image::RgbImage::from_pixel(
        DETECTION_TARGET_SIZE,
        DETECTION_TARGET_SIZE,
        image::Rgb([255, 255, 255]),
    );
    image::imageops::replace(&mut canvas, &resized, pad_x.into(), pad_y.into());

    let meta = ImageMeta {
        orig_width: orig_w,
        orig_height: orig_h,
        resized_width: DETECTION_TARGET_SIZE,
        resized_height: DETECTION_TARGET_SIZE,
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
    let ratio = crop.width() as f32 / crop.height() as f32;
    let target_w = (target_h as f32 * ratio).round() as u32;
    let target_w = target_w.clamp(RECOGNITION_MIN_WIDTH, RECOGNITION_MAX_WIDTH);

    let resized = resize_to_rgb(crop, target_w, target_h);
    normalize_rgb_to_nchw(&resized)
}

pub(crate) fn denormalize_detection_channel(channel: usize, value: f32) -> f32 {
    (value * DETECTION_STD[channel] + DETECTION_MEAN[channel]).clamp(0.0, 1.0)
}

/// 从检测框裁剪图片区域
pub fn crop_text_region(img: &DynamicImage, bbox: &BoundingBox) -> Result<DynamicImage, OcrError> {
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
    let pad_y = (rect_h * TEXT_REGION_VERTICAL_PAD_RATIO)
        .clamp(TEXT_REGION_MIN_PAD, TEXT_REGION_MAX_VERTICAL_PAD);

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
    let rgb = img.to_rgb8();
    image::imageops::resize(&rgb, width, height, image::imageops::FilterType::Lanczos3)
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
        let (data, meta) = preprocess_for_detection(&img);

        assert_eq!(meta.resized_width, DETECTION_TARGET_SIZE);
        assert_eq!(meta.resized_height, DETECTION_TARGET_SIZE);
        assert_eq!(meta.content_width, DETECTION_TARGET_SIZE);
        assert_eq!(meta.content_height, DETECTION_TARGET_SIZE);
        assert_eq!(meta.pad_x, 0);
        assert_eq!(meta.pad_y, 0);
        assert_eq!(
            data.len(),
            3 * DETECTION_TARGET_SIZE as usize * DETECTION_TARGET_SIZE as usize
        );

        let plane = DETECTION_TARGET_SIZE as usize * DETECTION_TARGET_SIZE as usize;
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
        let (_data, meta) = preprocess_for_detection(&img);

        assert_eq!(meta.resized_width, DETECTION_TARGET_SIZE);
        assert_eq!(meta.resized_height, DETECTION_TARGET_SIZE);
        assert_eq!(meta.content_width, DETECTION_TARGET_SIZE);
        assert_eq!(meta.content_height, DETECTION_TARGET_SIZE / 2);
        assert_eq!(meta.pad_x, 0);
        assert_eq!(meta.pad_y, DETECTION_TARGET_SIZE / 4);
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
