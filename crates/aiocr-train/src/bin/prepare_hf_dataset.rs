use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use aiocr_train::data::dataset::OcrDataset;
use serde::Deserialize;

#[derive(Debug)]
struct Args {
    dataset: String,
    split: String,
    offset: usize,
    rows: usize,
    out: PathBuf,
    dict: PathBuf,
    min_text_chars: usize,
}

#[derive(Debug, Deserialize)]
struct RowsResponse {
    rows: Vec<HfRow>,
}

#[derive(Debug, Deserialize)]
struct HfRow {
    row_idx: usize,
    row: HfRowData,
}

#[derive(Debug, Deserialize)]
struct HfRowData {
    image: HfImage,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    texts: Vec<String>,
    #[serde(default)]
    bboxes: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct HfImage {
    src: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    std::fs::create_dir_all(args.out.join("images"))?;

    let allowed_chars = load_allowed_chars(&args.dict)?;
    let url = format!(
        "https://datasets-server.huggingface.co/rows?dataset={}&config=default&split={}&offset={}&length={}",
        encode_dataset(&args.dataset),
        args.split,
        args.offset,
        args.rows,
    );
    let body = download_string(&url)?;
    let response: RowsResponse = serde_json::from_str(&body)?;

    let mut labels = String::new();
    let mut written = 0usize;
    let mut skipped = 0usize;

    for row in response.rows {
        let bytes = download_bytes(&row.row.image.src)?;
        let image = image::load_from_memory(&bytes)?;

        if row.row.bboxes.is_empty() {
            let Some(text) = row.row.text.as_deref().or(row.row.label.as_deref()) else {
                skipped += 1;
                continue;
            };
            let Some(label) = normalize_label(text, &allowed_chars, args.min_text_chars) else {
                skipped += 1;
                continue;
            };

            let filename = format!("{}_{}.jpg", args.split, row.row_idx);
            image.save(args.out.join("images").join(&filename))?;
            labels.push_str(&format!("{filename}\t{label}\n"));
            written += 1;
            continue;
        }

        for (region_index, bbox) in row.row.bboxes.iter().enumerate() {
            let Some(text) = row.row.texts.get(region_index) else {
                skipped += 1;
                continue;
            };
            let Some(label) = normalize_label(text, &allowed_chars, args.min_text_chars) else {
                skipped += 1;
                continue;
            };
            let Some(crop) = crop_bbox(&image, bbox) else {
                skipped += 1;
                continue;
            };

            let filename = format!("{}_{}_{}.jpg", args.split, row.row_idx, region_index);
            crop.save(args.out.join("images").join(&filename))?;
            labels.push_str(&format!("{filename}\t{label}\n"));
            written += 1;
        }
    }

    std::fs::write(args.out.join("labels.txt"), labels)?;
    println!(
        "dataset={} split={} written={} skipped={} out={}",
        args.dataset,
        args.split,
        written,
        skipped,
        args.out.display()
    );

    Ok(())
}

impl Args {
    fn parse() -> Result<Self, String> {
        let values = parse_key_values(std::env::args().skip(1).collect());
        Ok(Self {
            dataset: values
                .get("dataset")
                .cloned()
                .unwrap_or_else(|| "Yesianrohn/OCR-Data".to_string()),
            split: values
                .get("split")
                .cloned()
                .unwrap_or_else(|| "SCUT_HCCDoc".to_string()),
            offset: parse_usize(&values, "offset", 0)?,
            rows: parse_usize(&values, "rows", 20)?,
            out: PathBuf::from(
                values
                    .get("out")
                    .cloned()
                    .unwrap_or_else(|| "dataset/hf-scut-hccdoc".to_string()),
            ),
            dict: PathBuf::from(
                values
                    .get("dict")
                    .cloned()
                    .unwrap_or_else(|| "models/ppocr_keys_v1.txt".to_string()),
            ),
            min_text_chars: parse_usize(&values, "min-text-chars", 1)?,
        })
    }
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

fn parse_usize(
    values: &HashMap<String, String>,
    key: &str,
    default: usize,
) -> Result<usize, String> {
    values
        .get(key)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|err| format!("参数 --{key}={value} 不是有效数字: {err}"))
        })
        .unwrap_or(Ok(default))
}

fn load_allowed_chars(path: &Path) -> Result<HashSet<char>, Box<dyn std::error::Error>> {
    Ok(OcrDataset::build_char_map(path)?
        .keys()
        .copied()
        .collect::<HashSet<_>>())
}

fn download_bytes(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let output = curl_output(url, 120)?;
    Ok(output)
}

fn download_string(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(String::from_utf8(curl_output(url, 60)?)?)
}

fn curl_output(url: &str, timeout_secs: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let output = Command::new("curl")
        .arg("-L")
        .arg("--fail")
        .arg("--silent")
        .arg("--show-error")
        .arg("--max-time")
        .arg(timeout_secs.to_string())
        .arg(url)
        .output()?;

    if output.status.success() {
        return Ok(output.stdout);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!("curl 下载失败: {stderr}").into())
}

fn normalize_label(
    text: &str,
    allowed_chars: &HashSet<char>,
    min_text_chars: usize,
) -> Option<String> {
    let text = repair_mojibake(text);
    let label = text
        .chars()
        .filter(|ch| allowed_chars.contains(ch))
        .collect::<String>();

    (label.chars().count() >= min_text_chars).then_some(label)
}

fn repair_mojibake(text: &str) -> String {
    let Some(repaired) = cp1252_mojibake_to_utf8(text) else {
        return text.to_string();
    };

    if cjk_score(&repaired) > cjk_score(text) {
        repaired
    } else {
        text.to_string()
    }
}

fn cp1252_mojibake_to_utf8(text: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        bytes.push(cp1252_byte(ch)?);
    }
    String::from_utf8(bytes).ok()
}

fn cp1252_byte(ch: char) -> Option<u8> {
    if (ch as u32) <= 0xff {
        return Some(ch as u8);
    }

    match ch {
        '\u{20ac}' => Some(0x80),
        '\u{201a}' => Some(0x82),
        '\u{0192}' => Some(0x83),
        '\u{201e}' => Some(0x84),
        '\u{2026}' => Some(0x85),
        '\u{2020}' => Some(0x86),
        '\u{2021}' => Some(0x87),
        '\u{02c6}' => Some(0x88),
        '\u{2030}' => Some(0x89),
        '\u{0160}' => Some(0x8a),
        '\u{2039}' => Some(0x8b),
        '\u{0152}' => Some(0x8c),
        '\u{017d}' => Some(0x8e),
        '\u{2018}' => Some(0x91),
        '\u{2019}' => Some(0x92),
        '\u{201c}' => Some(0x93),
        '\u{201d}' => Some(0x94),
        '\u{2022}' => Some(0x95),
        '\u{2013}' => Some(0x96),
        '\u{2014}' => Some(0x97),
        '\u{02dc}' => Some(0x98),
        '\u{2122}' => Some(0x99),
        '\u{0161}' => Some(0x9a),
        '\u{203a}' => Some(0x9b),
        '\u{0153}' => Some(0x9c),
        '\u{017e}' => Some(0x9e),
        '\u{0178}' => Some(0x9f),
        _ => None,
    }
}

fn cjk_score(text: &str) -> usize {
    text.chars()
        .filter(|ch| {
            matches!(
                *ch as u32,
                0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff
            )
        })
        .count()
}

fn crop_bbox(image: &image::DynamicImage, bbox: &[f64]) -> Option<image::DynamicImage> {
    if bbox.len() < 4 {
        return None;
    }

    let width = image.width() as f64;
    let height = image.height() as f64;
    let left = bbox[0].min(bbox[2]).clamp(0.0, width - 1.0);
    let top = bbox[1].min(bbox[3]).clamp(0.0, height - 1.0);
    let right = bbox[0].max(bbox[2]).clamp(left + 1.0, width);
    let bottom = bbox[1].max(bbox[3]).clamp(top + 1.0, height);

    let crop_width = (right - left).round() as u32;
    let crop_height = (bottom - top).round() as u32;
    if crop_width < 2 || crop_height < 2 {
        return None;
    }

    Some(image.crop_imm(
        left.round() as u32,
        top.round() as u32,
        crop_width,
        crop_height,
    ))
}

fn encode_dataset(dataset: &str) -> String {
    dataset.replace('/', "%2F")
}
