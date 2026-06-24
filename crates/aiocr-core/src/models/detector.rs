use image::DynamicImage;

use crate::Detector;
use crate::config::{DetectionParams, DetectionResize};
use crate::error::OcrError;
use crate::postprocess::{DbPostprocessConfig, db_postprocess};
use crate::preprocess::{
    DETECTION_BURN_SQUARE_SIZE, denormalize_detection_channel, preprocess_for_detection,
};
use crate::types::BoundingBox;

#[cfg(aiocr_has_det)]
struct BurnDetectorRuntime {
    device: crate::models::BurnDevice,
    model: crate::models::det_generated::Model<crate::models::BurnBackend>,
}

#[cfg(aiocr_has_det)]
impl BurnDetectorRuntime {
    fn load() -> Result<Self, OcrError> {
        let device = crate::models::default_device();
        let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::models::det_generated::Model::<crate::models::BurnBackend>::from_embedded(
                &device,
            )
        }))
        .map_err(|_| OcrError::Inference("加载内嵌检测模型失败".to_string()))?;

        Ok(Self { device, model })
    }
}

/// 文本检测器（DBNet）
///
/// 当 ONNX 模型可用时，优先使用 burn-onnx 生成的纯 Burn 模型。
/// 在模型未生成时回退到纯 Rust 启发式检测。
pub struct TextDetector {
    /// 检测/后处理参数
    params: DetectionParams,
    #[cfg(aiocr_has_det)]
    runtime: std::sync::Mutex<Option<BurnDetectorRuntime>>,
}

impl TextDetector {
    pub fn generated_model_available() -> bool {
        crate::models::has_generated_detector()
    }

    pub fn new(params: DetectionParams) -> Result<Self, OcrError> {
        Ok(Self {
            params,
            #[cfg(aiocr_has_det)]
            runtime: std::sync::Mutex::new(None),
        })
    }

    /// 检测图片中的文本区域
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<(BoundingBox, f32)>, OcrError> {
        // Burn 内嵌检测器编译期固定 512×512，强制走方形预处理（回退启发式同样可用）。
        let (input_data, meta) =
            preprocess_for_detection(img, DetectionResize::square(DETECTION_BURN_SQUARE_SIZE));
        let h = meta.resized_height as usize;
        let w = meta.resized_width as usize;
        let prob_map = self.run_inference(&input_data, h, w)?;

        let boxes = db_postprocess(
            &prob_map,
            h,
            w,
            DbPostprocessConfig {
                threshold: self.params.threshold,
                box_threshold: self.params.box_threshold,
                max_candidates: self.params.max_candidates,
                unclip_ratio: self.params.unclip_ratio,
                min_box_area: self.params.min_box_area,
                box_mode: self.params.box_mode,
                meta: &meta,
            },
        );

        Ok(boxes)
    }

    fn run_inference(&self, input: &[f32], h: usize, w: usize) -> Result<Vec<f32>, OcrError> {
        #[cfg(aiocr_has_det)]
        {
            self.run_generated_inference(input, h, w)
        }

        #[cfg(not(aiocr_has_det))]
        {
            self.run_fallback_inference(input, h, w)
        }
    }

    #[cfg(aiocr_has_det)]
    fn run_generated_inference(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, OcrError> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|err| OcrError::Inference(format!("锁定检测模型状态失败: {err}")))?;
        if runtime.is_none() {
            *runtime = Some(BurnDetectorRuntime::load()?);
        }
        let runtime = runtime.as_ref().expect("runtime just initialized");

        let tensor = crate::models::nchw_tensor(input, [1, 3, h, w], &runtime.device);
        let output = runtime.model.forward(tensor);
        let dims = output.dims();
        let data = crate::models::tensor_to_vec(output)?;

        if dims[0] != 1 || dims[1] == 0 {
            return Err(OcrError::Inference(format!(
                "检测模型输出形状异常: expected [1, C, H, W], got {:?}",
                dims
            )));
        }

        let out_h = dims[2];
        let out_w = dims[3];
        let plane = out_h * out_w;
        if data.len() < plane {
            return Err(OcrError::Inference(format!(
                "检测模型输出数据不足: len={}, required={plane}, shape={dims:?}",
                data.len()
            )));
        }

        let mut prob_map = data[..plane].to_vec();
        if out_h != h || out_w != w {
            tracing::debug!(
                "检测模型输出尺寸 {}x{} 与预处理尺寸 {}x{} 不一致，执行概率图重采样",
                out_w,
                out_h,
                w,
                h
            );
            prob_map = resize_prob_map(&prob_map, out_w, out_h, w, h);
        }

        Ok(prob_map)
    }

    /// 启发式后备推理，用于无真实模型时维持可用性。
    #[cfg_attr(aiocr_has_det, allow(dead_code))]
    fn run_fallback_inference(
        &self,
        input: &[f32],
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, OcrError> {
        let plane = h * w;
        if input.len() != plane * 3 {
            return Err(OcrError::Inference(format!(
                "检测输入尺寸不匹配: got {}, expected {}",
                input.len(),
                plane * 3
            )));
        }

        let mut gray = vec![0.0f32; plane];
        for idx in 0..plane {
            let r = denormalize_detection_channel(0, input[idx]);
            let g = denormalize_detection_channel(1, input[plane + idx]);
            let b = denormalize_detection_channel(2, input[plane * 2 + idx]);
            gray[idx] = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0);
        }

        let background_is_light = gray.iter().sum::<f32>() / plane as f32 > 0.5;
        let mut prob_map = vec![0.0f32; plane];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let ink = if background_is_light {
                    1.0 - gray[idx]
                } else {
                    gray[idx]
                };

                let mut contrast_sum = 0.0f32;
                let mut contrast_count = 0usize;
                for (nx, ny) in neighbors(x, y, w, h) {
                    let nidx = ny * w + nx;
                    contrast_sum += (gray[idx] - gray[nidx]).abs();
                    contrast_count += 1;
                }

                let contrast = if contrast_count > 0 {
                    contrast_sum / contrast_count as f32
                } else {
                    0.0
                };
                prob_map[idx] = (ink * 0.75 + contrast * 0.25).clamp(0.0, 1.0);
            }
        }

        Ok(prob_map)
    }
}

impl Detector for TextDetector {
    fn detect(&self, img: &DynamicImage) -> Result<Vec<(BoundingBox, f32)>, OcrError> {
        TextDetector::detect(self, img)
    }

    fn name(&self) -> &str {
        if Self::generated_model_available() {
            "pp-ocrv5-det"
        } else {
            "dbnet-fallback"
        }
    }
}

#[cfg_attr(not(aiocr_has_det), allow(dead_code))]
fn resize_prob_map(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }

    let mut dst = vec![0.0f32; dst_w * dst_h];
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        let sy = (y as f32 + 0.5) * scale_y - 0.5;
        let y0 = sy.floor().clamp(0.0, (src_h - 1) as f32) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = (sy - y0 as f32).clamp(0.0, 1.0);

        for x in 0..dst_w {
            let sx = (x as f32 + 0.5) * scale_x - 0.5;
            let x0 = sx.floor().clamp(0.0, (src_w - 1) as f32) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = (sx - x0 as f32).clamp(0.0, 1.0);

            let top = src[y0 * src_w + x0] * (1.0 - wx) + src[y0 * src_w + x1] * wx;
            let bottom = src[y1 * src_w + x0] * (1.0 - wx) + src[y1 * src_w + x1] * wx;
            dst[y * dst_w + x] = top * (1.0 - wy) + bottom * wy;
        }
    }

    dst
}

#[cfg_attr(aiocr_has_det, allow(dead_code))]
fn neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let x = x as isize;
    let y = y as isize;
    const OFFSETS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    OFFSETS.into_iter().filter_map(move |(dx, dy)| {
        let nx = x + dx;
        let ny = y + dy;
        (nx >= 0 && ny >= 0 && nx < width as isize && ny < height as isize)
            .then_some((nx as usize, ny as usize))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_finds_dark_region_on_light_background() {
        let mut image = image::GrayImage::from_pixel(160, 96, image::Luma([255]));
        for y in 24..56 {
            for x in 24..120 {
                image.put_pixel(x, y, image::Luma([0]));
            }
        }

        let detector = TextDetector::new(DetectionParams {
            box_threshold: 0.4,
            max_candidates: 32,
            unclip_ratio: 1.2,
            ..Default::default()
        })
        .unwrap();
        let (input, meta) =
            preprocess_for_detection(&DynamicImage::ImageLuma8(image), DetectionResize::square(512));
        let result = detector
            .run_fallback_inference(
                &input,
                meta.resized_height as usize,
                meta.resized_width as usize,
            )
            .unwrap();

        assert_eq!(
            result.len(),
            meta.resized_width as usize * meta.resized_height as usize
        );
        assert!(result.iter().any(|value| *value > 0.7));
    }

    #[cfg(aiocr_has_det)]
    #[test]
    fn test_generated_detector_smoke() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                    96,
                    64,
                    image::Rgb([255, 255, 255]),
                ));
                let detector = TextDetector::new(DetectionParams {
            box_threshold: 0.4,
            max_candidates: 32,
            unclip_ratio: 1.2,
            ..Default::default()
        })
        .unwrap();
                let result = detector.detect(&image).unwrap();
                assert!(result.len() <= 32);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
