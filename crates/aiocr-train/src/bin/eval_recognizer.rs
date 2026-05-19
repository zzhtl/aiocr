use std::collections::HashMap;
use std::path::PathBuf;

use aiocr_core::decode::CtcDecoder;
use aiocr_core::models::recognizer::TextRecognizer;
use aiocr_core::Recognizer;
use aiocr_train::AiModelInfo;
use aiocr_train::data::dataset::OcrDataset;

#[derive(Debug)]
struct Args {
    dataset: PathBuf,
    model: Option<PathBuf>,
    dict: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let dataset = OcrDataset::from_dir_raw(&args.dataset)?;
    let recognizer = build_recognizer(&args)?;

    let mut exact = 0usize;
    let mut total = 0usize;
    let mut edit_distance = 0usize;
    let mut target_chars = 0usize;

    for item in dataset.items() {
        let image = image::open(&item.image_path)?;
        let (prediction, confidence) = recognizer.recognize(&image)?;
        let target = item.label_text.trim();
        let prediction = prediction.trim();

        total += 1;
        if prediction == target {
            exact += 1;
        }
        edit_distance += levenshtein_chars(prediction, target);
        target_chars += target.chars().count();

        println!(
            "sample={} conf={:.4} pred={} target={}",
            item.image_path.display(),
            confidence,
            prediction,
            target,
        );
    }

    let line_accuracy = if total == 0 {
        0.0
    } else {
        exact as f32 / total as f32
    };
    let char_accuracy = if target_chars == 0 {
        0.0
    } else {
        1.0 - edit_distance as f32 / target_chars as f32
    };

    println!(
        "summary samples={} line_accuracy={:.4} char_accuracy={:.4} edit_distance={} target_chars={}",
        total,
        line_accuracy,
        char_accuracy.max(0.0),
        edit_distance,
        target_chars,
    );

    Ok(())
}

impl Args {
    fn parse() -> Result<Self, String> {
        let values = parse_key_values(std::env::args().skip(1).collect());
        Ok(Self {
            dataset: PathBuf::from(
                values
                    .get("dataset")
                    .cloned()
                    .unwrap_or_else(|| "dataset/hf-scut-hccdoc".to_string()),
            ),
            model: values.get("model").map(PathBuf::from),
            dict: PathBuf::from(
                values
                    .get("dict")
                    .cloned()
                    .unwrap_or_else(|| "models/ppocr_keys_v1.txt".to_string()),
            ),
        })
    }
}

fn build_recognizer(args: &Args) -> Result<Box<dyn Recognizer>, Box<dyn std::error::Error>> {
    if let Some(model_dir) = &args.model {
        let model = AiModelInfo::load_from_dir(model_dir)?;
        let decoder = CtcDecoder::from_dict_file(&model.dict_path)?;
        return Ok(Box::new(TextRecognizer::from_burnpack_file(
            decoder,
            model.weights_path,
            model.name,
        )?));
    }

    let decoder = CtcDecoder::from_dict_file(&args.dict)?;
    Ok(Box::new(TextRecognizer::new(decoder)?))
}

fn parse_key_values(args: Vec<String>) -> HashMap<String, String> {
    let mut values = HashMap::new();
    let mut index = 0usize;
    while index < args.len() {
        let arg = &args[index];
        if let Some(key) = arg.strip_prefix("--") {
            if let Some((key, value)) = key.split_once('=') {
                values.insert(key.to_string(), value.to_string());
            } else if let Some(value) = args.get(index + 1) {
                values.insert(key.to_string(), value.clone());
                index += 1;
            }
        }
        index += 1;
    }
    values
}

fn levenshtein_chars(left: &str, right: &str) -> usize {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0usize; right.len() + 1];

    for (i, left_char) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, right_char) in right.iter().enumerate() {
            let substitution = previous[j] + usize::from(left_char != right_char);
            let insertion = current[j] + 1;
            let deletion = previous[j + 1] + 1;
            current[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[right.len()]
}
