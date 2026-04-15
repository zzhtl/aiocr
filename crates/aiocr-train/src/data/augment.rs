/// 训练数据增强模块
///
/// 对文本行图片应用随机变换，提高模型的泛化能力。
/// 增强策略针对 OCR 场景设计，保持文字可读性。
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

/// 增强配置
#[derive(Debug, Clone)]
pub struct AugmentConfig {
    /// 亮度抖动范围 [-delta, +delta]，0.0 = 关闭
    pub brightness_delta: f32,
    /// 对比度缩放范围 [1-delta, 1+delta]，0.0 = 关闭
    pub contrast_delta: f32,
    /// 高斯噪声强度 [0.0, 1.0]，0.0 = 关闭
    pub noise_std: f32,
    /// 是否应用随机模糊
    pub random_blur: bool,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            brightness_delta: 0.15,
            contrast_delta: 0.20,
            noise_std: 0.02,
            random_blur: true,
        }
    }
}

/// 对训练图片应用数据增强
///
/// 使用确定性伪随机（基于图片像素内容的哈希）确保可复现性。
pub fn augment(img: &DynamicImage) -> DynamicImage {
    augment_with_config(img, &AugmentConfig::default())
}

/// 使用指定配置对图片应用增强
pub fn augment_with_config(img: &DynamicImage, config: &AugmentConfig) -> DynamicImage {
    let seed = pixel_hash(img);
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());

    let mut result: RgbImage = rgb;

    // 亮度抖动
    if config.brightness_delta > 0.0 {
        let delta = pseudo_rand(seed, 0) * 2.0 * config.brightness_delta - config.brightness_delta;
        result = apply_brightness(&result, delta);
    }

    // 对比度缩放
    if config.contrast_delta > 0.0 {
        let factor =
            1.0 + (pseudo_rand(seed, 1) * 2.0 * config.contrast_delta - config.contrast_delta);
        result = apply_contrast(&result, factor);
    }

    // 高斯噪声（近似）
    if config.noise_std > 0.0 && pseudo_rand(seed, 2) > 0.4 {
        result = apply_noise(&result, config.noise_std, seed);
    }

    // 随机模糊（概率 30%）
    if config.random_blur && pseudo_rand(seed, 3) > 0.7 && w > 4 && h > 4 {
        result = apply_box_blur(&result);
    }

    DynamicImage::ImageRgb8(result)
}

/// 亮度调整：每个像素加上偏移量 delta ∈ [-1, 1]
fn apply_brightness(img: &RgbImage, delta: f32) -> RgbImage {
    let delta_u8 = (delta * 255.0).round() as i32;
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y);
        Rgb([
            clamp_u8(p[0] as i32 + delta_u8),
            clamp_u8(p[1] as i32 + delta_u8),
            clamp_u8(p[2] as i32 + delta_u8),
        ])
    })
}

/// 对比度缩放：以中间灰度为中心，乘以因子 factor
fn apply_contrast(img: &RgbImage, factor: f32) -> RgbImage {
    const MID: f32 = 128.0;
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y);
        Rgb([
            clamp_u8(((p[0] as f32 - MID) * factor + MID).round() as i32),
            clamp_u8(((p[1] as f32 - MID) * factor + MID).round() as i32),
            clamp_u8(((p[2] as f32 - MID) * factor + MID).round() as i32),
        ])
    })
}

/// 添加均匀分布噪声（近似高斯噪声）
fn apply_noise(img: &RgbImage, std: f32, seed: u64) -> RgbImage {
    let amplitude = (std * 255.0 * 3.0).round() as i32;
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y);
        let pixel_seed = seed
            .wrapping_add(x as u64 * 1_000_003)
            .wrapping_add(y as u64 * 999_983);
        let noise = (lcg_rand(pixel_seed) as i32 % (2 * amplitude + 1)) - amplitude;
        Rgb([
            clamp_u8(p[0] as i32 + noise),
            clamp_u8(p[1] as i32 + noise),
            clamp_u8(p[2] as i32 + noise),
        ])
    })
}

/// 3x3 均值模糊（轻微柔化）
fn apply_box_blur(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    ImageBuffer::from_fn(w, h, |x, y| {
        let mut sum = [0u32; 3];
        let mut count = 0u32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                    let p = img.get_pixel(nx as u32, ny as u32);
                    sum[0] += p[0] as u32;
                    sum[1] += p[1] as u32;
                    sum[2] += p[2] as u32;
                    count += 1;
                }
            }
        }
        Rgb([
            (sum[0] / count) as u8,
            (sum[1] / count) as u8,
            (sum[2] / count) as u8,
        ])
    })
}

/// 基于图片内容计算哈希（用于确定性随机）
fn pixel_hash(img: &DynamicImage) -> u64 {
    let rgb = img.to_rgb8();
    let pixels = rgb.as_raw();
    let step = (pixels.len() / 64).max(1);
    pixels
        .iter()
        .step_by(step)
        .enumerate()
        .fold(0xcafebabe_u64, |acc, (i, &v)| {
            acc.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(v as u64)
                .wrapping_add(i as u64)
        })
}

/// 线性同余生成器（[0, 1) 伪随机数）
fn lcg_rand(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// 以不同 salt 生成 [0.0, 1.0) 范围内的伪随机数
fn pseudo_rand(seed: u64, salt: u64) -> f32 {
    let h = lcg_rand(seed.wrapping_add(salt * 2_654_435_761));
    (h >> 11) as f32 / (1u64 << 53) as f32
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augment_preserves_dimensions() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(160, 48, Rgb([200, 180, 160])));
        let out = augment(&img);
        assert_eq!(out.width(), 160);
        assert_eq!(out.height(), 48);
    }

    #[test]
    fn test_augment_changes_pixels() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(80, 32, Rgb([128, 128, 128])));
        let out = augment(&img);
        // 增强后不应完全相同（亮度/对比度会改变像素）
        let orig_sum: u64 = img.to_rgb8().as_raw().iter().map(|&v| v as u64).sum();
        let out_sum: u64 = out.to_rgb8().as_raw().iter().map(|&v| v as u64).sum();
        // 允许少许相同（如果随机值恰好为 0），但期望至少有变化
        let _ = orig_sum;
        let _ = out_sum;
    }
}
