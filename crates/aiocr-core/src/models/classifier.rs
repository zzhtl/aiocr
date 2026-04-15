use image::DynamicImage;

use crate::error::OcrError;
use crate::preprocess::preprocess_for_classification;
use crate::types::TextDirection;

#[cfg(aiocr_has_cls)]
struct BurnClassifierRuntime {
    device: crate::models::BurnDevice,
    model: crate::models::cls_generated::Model<crate::models::BurnBackend>,
}

#[cfg(aiocr_has_cls)]
impl BurnClassifierRuntime {
    fn load() -> Result<Self, OcrError> {
        let device = crate::models::default_device();
        let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::models::cls_generated::Model::<crate::models::BurnBackend>::from_embedded(
                &device,
            )
        }))
        .map_err(|_| OcrError::Inference("加载内嵌方向分类模型失败".to_string()))?;

        Ok(Self { device, model })
    }
}

/// 文本方向分类器
///
/// 判断文本行是否需要旋转 180 度
pub struct DirectionClassifier {
    /// 分类置信度阈值
    threshold: f32,
    #[cfg(aiocr_has_cls)]
    runtime: std::sync::Mutex<Option<BurnClassifierRuntime>>,
}

impl DirectionClassifier {
    pub fn generated_model_available() -> bool {
        crate::models::has_generated_classifier()
    }

    pub fn new(threshold: f32) -> Result<Self, OcrError> {
        Ok(Self {
            threshold,
            #[cfg(aiocr_has_cls)]
            runtime: std::sync::Mutex::new(None),
        })
    }

    /// 分类文本方向，返回是否需要旋转 180 度
    pub fn classify(&self, crop: &DynamicImage) -> Result<(TextDirection, bool), OcrError> {
        let direction = if crop.height() > crop.width().saturating_mul(5) / 4 {
            TextDirection::Vertical
        } else {
            TextDirection::Horizontal
        };

        #[cfg(aiocr_has_cls)]
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|err| OcrError::Inference(format!("锁定方向分类模型状态失败: {err}")))?;
            if runtime.is_none() {
                *runtime = Some(BurnClassifierRuntime::load()?);
            }
            let runtime = runtime.as_ref().expect("runtime just initialized");

            let input = preprocess_for_classification(crop);
            let tensor = crate::models::nchw_tensor(&input, [1, 3, 48, 192], &runtime.device);
            let output = runtime.model.forward(tensor);
            let dims = output.dims();
            let data = crate::models::tensor_to_vec(output)?;

            if dims != [1, 2] || data.len() != 2 {
                return Err(OcrError::Inference(format!(
                    "方向分类模型输出形状异常: expected [1, 2], got {:?}",
                    dims
                )));
            }

            let (label, confidence) = if data[0] >= data[1] {
                (0usize, data[0])
            } else {
                (1usize, data[1])
            };
            let need_rotate = direction == TextDirection::Horizontal
                && label == 1
                && confidence >= self.threshold.min(0.999);

            Ok((direction, need_rotate))
        }

        #[cfg(not(aiocr_has_cls))]
        {
            let gray = crop.to_luma8();
            let half = (gray.height() / 2).max(1);
            let mut top_half = 0.0f32;
            let mut bottom_half = 0.0f32;

            for (y, pixel) in gray.enumerate_rows() {
                let row_ink = pixel
                    .map(|(_, _, value)| 1.0 - (value.0[0] as f32 / 255.0))
                    .sum::<f32>();
                if y < half {
                    top_half += row_ink;
                } else {
                    bottom_half += row_ink;
                }
            }

            let total = (top_half + bottom_half).max(f32::EPSILON);
            let confidence = ((bottom_half - top_half).abs() / total).clamp(0.0, 1.0);
            let need_rotate = direction == TextDirection::Horizontal
                && bottom_half > top_half
                && confidence >= self.threshold.min(0.98);

            Ok((direction, need_rotate))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(aiocr_has_cls)]
    #[test]
    fn test_generated_classifier_smoke() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                    192,
                    48,
                    image::Rgb([255, 255, 255]),
                ));
                let classifier = DirectionClassifier::new(0.9).unwrap();
                let (direction, _need_rotate) = classifier.classify(&image).unwrap();
                assert_eq!(direction, TextDirection::Horizontal);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
