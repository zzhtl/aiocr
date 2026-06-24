use aiocr_core::config::OcrConfig;
use aiocr_core::decode::CtcDecoder;
use aiocr_core::models::classifier::DirectionClassifier;
use aiocr_core::models::detector::TextDetector;
use aiocr_core::models::onnx_backend::{OnnxDetector, OnnxRecognizer};
use aiocr_core::models::recognizer::TextRecognizer;
use aiocr_core::{Detector, OcrEngine, Recognizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let image_path = args
        .next()
        .ok_or("usage: cargo run -p aiocr-core --example ocr_probe -- <image-path> [repeat]")?;
    let repeat = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1)
        .max(1);

    let models_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut config = OcrConfig::default();
    config.weights_dir = models_dir.clone();
    config.dict_path = models_dir.join("ppocr_keys_v1.txt");

    let image = image::open(&image_path)?;
    let det_backend = std::env::var("AIOCR_DETECTOR").unwrap_or_else(|_| "onnx".to_string());
    let rec_backend = std::env::var("AIOCR_RECOGNIZER").unwrap_or_else(|_| "burn".to_string());
    let detector: Box<dyn Detector> = match det_backend.as_str() {
        "onnx" => Box::new(OnnxDetector::new(
            &models_dir.join("det.onnx"),
            config.detection_params(),
        )?),
        "burn" => Box::new(TextDetector::new(config.detection_params())?),
        other => return Err(format!("unsupported detector backend: {other}").into()),
    };
    let classifier = DirectionClassifier::new(config.cls_threshold)?;
    let decoder = CtcDecoder::from_dict_or_builtin(&config.dict_path)?;
    let recognizer: Box<dyn Recognizer> = match rec_backend.as_str() {
        "onnx" => Box::new(OnnxRecognizer::new(
            &models_dir.join("rec.onnx"),
            decoder,
            "onnx-rec",
        )?),
        "burn" => Box::new(TextRecognizer::new(decoder)?),
        other => return Err(format!("unsupported recognizer backend: {other}").into()),
    };
    let engine = OcrEngine::new(detector, classifier, recognizer);

    println!("image={image_path}");
    println!(
        "det_backend={det_backend} rec_backend={rec_backend} generated_det={} generated_cls={} generated_rec={}",
        TextDetector::generated_model_available(),
        DirectionClassifier::generated_model_available(),
        TextRecognizer::generated_model_available()
    );

    let mut total_ms_sum = 0u128;

    for run in 0..repeat {
        let result = engine.run(&image)?;
        total_ms_sum += result.elapsed_ms as u128;

        println!(
            "run[{run}] total_ms={} regions={}",
            result.elapsed_ms,
            result.regions.len()
        );
        for (index, region) in result.regions.iter().enumerate() {
            println!(
                "run[{run}] region[{index}] text={:?} confidence={:.4} bbox={:?}",
                region.text, region.confidence, region.bbox.points
            );
        }
        println!("run[{run}] full_text={:?}", result.full_text);
    }

    if repeat > 1 {
        println!("avg total_ms={}", total_ms_sum / repeat as u128);
    }

    Ok(())
}
