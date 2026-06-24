use image::DynamicImage;

use crate::config::OcrConfig;
use crate::decode::CtcDecoder;
use crate::error::OcrError;
use crate::log_ocr_stage_timings;
use crate::models::classifier::DirectionClassifier;
use crate::models::detector::TextDetector;
use crate::models::recognizer::TextRecognizer;
use crate::run_ocr_flow;
use crate::types::OcrResult;
use crate::{Detector, Recognizer};

/// OCR 管线：检测 → 分类 → 识别
pub struct OcrPipeline {
    detector: TextDetector,
    classifier: DirectionClassifier,
    recognizer: TextRecognizer,
    rec_score_threshold: f32,
}

impl OcrPipeline {
    /// 从配置创建 OCR 管线
    pub fn new(config: &OcrConfig) -> Result<Self, OcrError> {
        let detector = TextDetector::new(config.detection_params())?;

        let classifier = DirectionClassifier::new(config.cls_threshold)?;

        let decoder = CtcDecoder::from_dict_or_builtin(&config.dict_path)?;
        let recognizer = TextRecognizer::new(decoder)?;

        Ok(Self {
            detector,
            classifier,
            recognizer,
            rec_score_threshold: config.rec_score_threshold,
        })
    }

    /// 执行完整 OCR 流程
    pub fn run(&self, img: &DynamicImage) -> Result<OcrResult, OcrError> {
        let (result, timings) = run_ocr_flow(
            &self.detector,
            &self.classifier,
            &self.recognizer,
            img,
            self.rec_score_threshold,
        )?;
        log_ocr_stage_timings(
            self.detector.name(),
            self.recognizer.name(),
            result.regions.len(),
            &timings,
        );
        tracing::info!("OCR 完成，耗时 {}ms", result.elapsed_ms);
        Ok(result)
    }
}
