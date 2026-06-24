//! AIOCR Core - 纯 Rust OCR 推理引擎
//!
//! 默认优先使用 `models/` 下的 PP-OCRv5 Server 模型。
//! 检测优先走 ONNX，识别优先走 Burn；当对应后端不可用时再回退。
//! 同时支持通过 tract-onnx 直接加载外部 ONNX 文件。

pub mod config;
pub mod decode;
pub mod error;
pub mod models;
pub mod pipeline;
pub mod postprocess;
pub mod preprocess;
pub mod types;

use std::time::{Duration, Instant};

use image::DynamicImage;

use crate::config::OnnxModelConfig;
use crate::decode::CtcDecoder;
use crate::error::OcrError;
use crate::models::classifier::DirectionClassifier;
use crate::models::detector::TextDetector;
use crate::models::onnx_backend::{OnnxDetector, OnnxRecognizer};
use crate::models::recognizer::TextRecognizer;
use crate::types::{BoundingBox, OcrResult};

// 重导出核心类型
pub use config::{
    DetectionBoxMode, DetectionParams, DetectionResize, OcrConfig,
    OnnxModelConfig as OcrOnnxConfig, QualityPreset,
};
pub use error::OcrError as Error;
pub use pipeline::OcrPipeline;
pub use types::{OcrResult as Result, TextRegion};

/// 可插拔的文本识别器 trait
pub trait Recognizer: Send + Sync {
    /// 识别裁剪的文本行图片
    fn recognize(&self, crop: &DynamicImage) -> std::result::Result<(String, f32), OcrError>;
    /// 批量识别裁剪的文本行图片
    fn recognize_batch(
        &self,
        crops: &[DynamicImage],
    ) -> Vec<std::result::Result<(String, f32), OcrError>> {
        crops.iter().map(|crop| self.recognize(crop)).collect()
    }
    /// 识别器名称
    fn name(&self) -> &str;
}

/// 可插拔的文本检测器 trait
pub trait Detector: Send + Sync {
    /// 检测图片中的文本区域
    fn detect(&self, img: &DynamicImage) -> std::result::Result<Vec<(BoundingBox, f32)>, OcrError>;
    /// 检测器名称
    fn name(&self) -> &str;
}

/// OcrEngine: 支持热替换识别后端的 OCR 引擎
pub struct OcrEngine {
    detector: Box<dyn Detector>,
    classifier: DirectionClassifier,
    recognizer: Box<dyn Recognizer>,
    /// 识别置信度阈值，低于此值的区域不进入版式文本（0 = 不过滤）。
    rec_score_threshold: f32,
}

impl OcrEngine {
    /// 创建引擎
    pub fn new(
        detector: Box<dyn Detector>,
        classifier: DirectionClassifier,
        recognizer: Box<dyn Recognizer>,
    ) -> Self {
        Self {
            detector,
            classifier,
            recognizer,
            rec_score_threshold: 0.0,
        }
    }

    /// 设置识别置信度过滤阈值（0 = 不过滤）。
    pub fn with_rec_score_threshold(mut self, threshold: f32) -> Self {
        self.rec_score_threshold = threshold;
        self
    }

    /// 通过默认 OCR 配置创建默认引擎
    pub fn from_config(config: &OcrConfig) -> std::result::Result<Self, OcrError> {
        let default_onnx = OnnxModelConfig::from_dir(&config.weights_dir);
        let det_params = config.detection_params();
        let detector: Box<dyn Detector> = if let Some(det_path) = &default_onnx.det_path {
            tracing::info!("默认 OCR 检测使用 ONNX 模型: {}", det_path.display());
            Box::new(OnnxDetector::new(det_path, det_params)?)
        } else if TextDetector::generated_model_available() {
            tracing::warn!("models/ 下未找到 det.onnx，回退到 Burn 检测模型");
            Box::new(TextDetector::new(det_params)?)
        } else {
            tracing::warn!("检测 ONNX 与 Burn 模型都不可用，回退到内置检测器");
            Box::new(TextDetector::new(det_params)?)
        };

        let classifier = DirectionClassifier::new(config.cls_threshold)?;
        let dict_path = default_onnx
            .dict_path
            .as_deref()
            .unwrap_or(&config.dict_path);
        let recognizer: Box<dyn Recognizer> = if TextRecognizer::generated_model_available() {
            tracing::info!("默认 OCR 识别使用 Burn 模型");
            let decoder = CtcDecoder::from_dict_or_builtin(dict_path)?;
            Box::new(TextRecognizer::new(decoder)?)
        } else if let Some(rec_path) = &default_onnx.rec_path {
            tracing::warn!(
                "默认 Burn 识别模型不可用，回退到 ONNX 识别: {}",
                rec_path.display()
            );
            let decoder = CtcDecoder::from_dict_or_builtin(dict_path)?;
            Box::new(OnnxRecognizer::new(rec_path, decoder, "onnx-rec")?)
        } else {
            let decoder = CtcDecoder::from_dict_or_builtin(&config.dict_path)?;
            Box::new(TextRecognizer::new(decoder)?)
        };

        Ok(Self::new(detector, classifier, recognizer)
            .with_rec_score_threshold(config.rec_score_threshold))
    }

    /// 通过 ONNX 文件创建引擎（使用 tract-onnx 运行时）
    ///
    /// 支持任意 PaddleOCR 兼容的 ONNX 模型，无需重新编译。
    pub fn from_onnx(
        base_config: &OcrConfig,
        onnx: &OnnxModelConfig,
    ) -> std::result::Result<Self, OcrError> {
        // 检测器：优先使用 ONNX，否则回退到 Burn 内嵌
        let det_params = base_config.detection_params();
        let detector: Box<dyn Detector> = if let Some(det_path) = &onnx.det_path {
            Box::new(OnnxDetector::new(det_path, det_params)?)
        } else {
            tracing::warn!("ONNX 检测模型未配置，回退到 Burn 内嵌模型");
            Box::new(TextDetector::new(det_params)?)
        };

        // 方向分类器：始终使用 Burn 内嵌（cls 对效果影响有限）
        let classifier = DirectionClassifier::new(base_config.cls_threshold)?;

        // 识别器：优先使用 ONNX
        let dict_path = onnx.dict_path.as_deref().unwrap_or(&base_config.dict_path);
        let recognizer: Box<dyn Recognizer> = if let Some(rec_path) = &onnx.rec_path {
            let decoder = CtcDecoder::from_dict_or_builtin(dict_path)?;
            Box::new(OnnxRecognizer::new(rec_path, decoder, "onnx-rec")?)
        } else {
            tracing::warn!("ONNX 识别模型未配置，回退到 Burn 内嵌模型");
            let decoder = CtcDecoder::from_dict_or_builtin(&base_config.dict_path)?;
            Box::new(TextRecognizer::new(decoder)?)
        };

        Ok(Self::new(detector, classifier, recognizer)
            .with_rec_score_threshold(base_config.rec_score_threshold))
    }

    /// 执行 OCR
    pub fn run(&self, img: &DynamicImage) -> std::result::Result<OcrResult, OcrError> {
        let (result, timings) = run_ocr_flow(
            &*self.detector,
            &self.classifier,
            &*self.recognizer,
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

    /// 热替换识别后端
    pub fn swap_recognizer(&mut self, recognizer: Box<dyn Recognizer>) {
        tracing::info!("切换识别后端: {}", recognizer.name());
        self.recognizer = recognizer;
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct OcrStageTimings {
    detect: Duration,
    crop: Duration,
    classify: Duration,
    recognize: Duration,
    join: Duration,
}

pub(crate) fn run_ocr_flow<D, R>(
    detector: &D,
    classifier: &DirectionClassifier,
    recognizer: &R,
    img: &DynamicImage,
    rec_score_threshold: f32,
) -> std::result::Result<(OcrResult, OcrStageTimings), OcrError>
where
    D: Detector + ?Sized,
    R: Recognizer + ?Sized,
{
    let total_start = Instant::now();

    let detect_start = Instant::now();
    let detections = detector.detect(img)?;
    let detections = prepare_detections_for_recognition(&detections, img.width() as f32);
    let mut timings = OcrStageTimings {
        detect: detect_start.elapsed(),
        ..OcrStageTimings::default()
    };
    tracing::info!("检测到 {} 个文本区域", detections.len());

    let mut crop_infos = Vec::with_capacity(detections.len());
    let mut crops = Vec::with_capacity(detections.len());

    for (bbox, det_conf) in &detections {
        let crop_start = Instant::now();
        let crop = match preprocess::crop_text_region(img, bbox) {
            Ok(crop) => crop,
            Err(err) => {
                tracing::warn!("裁剪文本区域失败: {err}");
                timings.crop += crop_start.elapsed();
                continue;
            }
        };
        timings.crop += crop_start.elapsed();

        crop_infos.push((*bbox, *det_conf));
        crops.push(crop);
    }

    let classify_start = Instant::now();
    let classifications = match classifier.classify_batch(&crops) {
        Ok(classifications) => classifications,
        Err(err) => {
            tracing::warn!("批量方向分类失败，使用默认方向: {err}");
            vec![(types::TextDirection::Horizontal, false); crops.len()]
        }
    };
    timings.classify += classify_start.elapsed();

    let mut recognition_crops = Vec::with_capacity(crops.len());
    for (crop, (_, need_rotate)) in crops.into_iter().zip(classifications.iter().copied()) {
        recognition_crops.push(if need_rotate { crop.rotate180() } else { crop });
    }

    let recognize_start = Instant::now();
    let recognition_results = recognizer.recognize_batch(&recognition_crops);
    timings.recognize += recognize_start.elapsed();

    let mut regions = Vec::with_capacity(crop_infos.len());

    for (((bbox, det_conf), (direction, _)), recognition) in crop_infos
        .into_iter()
        .zip(classifications.into_iter())
        .zip(recognition_results.into_iter())
    {
        let (text, rec_conf) = match recognition {
            Ok(recognition) => recognition,
            Err(err) => {
                tracing::warn!("识别器 {} 处理区域失败: {err}", recognizer.name());
                continue;
            }
        };

        if text.trim().is_empty() {
            continue;
        }

        regions.push(types::TextRegion {
            bbox,
            confidence: det_conf * rec_conf,
            text,
            direction,
        });
    }

    let join_start = Instant::now();
    // 低置信度区域不进入版式文本，但仍保留在 regions 供 GUI 高亮/调试。
    let full_text = if rec_score_threshold > 0.0 {
        let filtered: Vec<types::TextRegion> = regions
            .iter()
            .filter(|region| region.confidence >= rec_score_threshold)
            .cloned()
            .collect();
        build_full_text(&filtered)
    } else {
        build_full_text(&regions)
    };
    timings.join = join_start.elapsed();

    let result = OcrResult {
        regions,
        full_text,
        elapsed_ms: total_start.elapsed().as_millis() as u64,
    };

    Ok((result, timings))
}

fn build_full_text(regions: &[types::TextRegion]) -> String {
    if regions.is_empty() {
        return String::new();
    }

    let grouped_lines = group_regions_by_line(regions);
    let reconstructed_lines = reconstruct_layout_lines(&grouped_lines);
    let numbering_start = detect_line_numbering_start(&reconstructed_lines, grouped_lines.len());

    let line_number_width = numbering_start.map_or(0, |start| {
        (start as usize + reconstructed_lines.len().saturating_sub(1))
            .to_string()
            .len()
    });
    let capacity = reconstructed_lines
        .iter()
        .map(|line| line.text.len() + line_number_width + 2)
        .sum::<usize>()
        + reconstructed_lines.len().saturating_sub(1);
    let mut full_text = String::with_capacity(capacity);

    for (index, line) in reconstructed_lines.iter().enumerate() {
        if index > 0 {
            full_text.push('\n');
        }

        let mut content = line.text.clone();
        if let Some(start) = numbering_start {
            let line_number = start + index as u32;
            push_right_aligned_number(&mut full_text, line_number, line_number_width);
            content = strip_matching_line_number_prefix(&content, line_number);
            if !content.trim().is_empty() {
                full_text.push_str("  ");
            }
        }

        full_text.push_str(content.trim_end());
    }

    full_text
}

/// 按文本框在原图中的相对位置重建可复制的版式文本。
///
/// `build_full_text` 更偏向连续阅读顺序；这个函数保留水平缩进和较大的纵向空白，
/// 用于表格、票据、截图等复杂格式的结果展示。
pub fn build_spatial_text(
    regions: &[types::TextRegion],
    image_width: f32,
    image_height: f32,
) -> String {
    if regions.is_empty() {
        return String::new();
    }

    let mut regions = regions.to_vec();
    regions.sort_by(|left, right| {
        left.bbox
            .top()
            .partial_cmp(&right.bbox.top())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.bbox
                    .left()
                    .partial_cmp(&right.bbox.left())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let lines = group_regions_by_line(&regions);
    let char_width = estimate_spatial_char_width(&regions, image_width);
    let line_pitch = estimate_spatial_line_pitch(&lines, image_height);
    let min_left = regions
        .iter()
        .map(|region| region.bbox.left())
        .fold(f32::MAX, f32::min)
        .max(0.0);

    let mut output = String::new();
    let mut previous_center_y: Option<f32> = None;

    for line in lines {
        if let Some(previous) = previous_center_y {
            let gap = (line_center_y(&line) - previous).max(0.0);
            let blank_lines = ((gap / line_pitch).round() as isize - 1)
                .max(0)
                .min(16) as usize;
            for _ in 0..blank_lines {
                output.push('\n');
            }
        }

        if !output.is_empty() {
            output.push('\n');
        }

        output.push_str(&build_spatial_line(&line, min_left, char_width));
        previous_center_y = Some(line_center_y(&line));
    }

    output
}

fn estimate_spatial_char_width(regions: &[types::TextRegion], image_width: f32) -> f32 {
    let mut samples = regions
        .iter()
        .filter_map(|region| {
            let chars = region.text.chars().count().max(1) as f32;
            let width = region.bbox.width() / chars;
            (width.is_finite() && width > 0.0).then_some(width)
        })
        .collect::<Vec<_>>();
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

    let median = if samples.is_empty() {
        10.0
    } else {
        samples[samples.len() / 2]
    };
    let max_columns = 160.0;
    let width_for_limit = if image_width.is_finite() && image_width > 0.0 {
        image_width / max_columns
    } else {
        1.0
    };

    median.max(width_for_limit).max(1.0)
}

fn estimate_spatial_line_pitch(
    lines: &[Vec<&types::TextRegion>],
    image_height: f32,
) -> f32 {
    let mut deltas = lines
        .windows(2)
        .filter_map(|pair| {
            let delta = line_center_y(&pair[1]) - line_center_y(&pair[0]);
            (delta.is_finite() && delta > 0.0).then_some(delta)
        })
        .collect::<Vec<_>>();
    deltas.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

    if !deltas.is_empty() {
        let sample_len = deltas.len().div_ceil(2);
        let sample = &deltas[..sample_len];
        return sample[sample.len() / 2].max(1.0);
    }

    let avg_height = lines
        .iter()
        .flat_map(|line| line.iter().map(|region| region.bbox.height()))
        .sum::<f32>()
        / lines.iter().map(Vec::len).sum::<usize>().max(1) as f32;

    avg_height
        .max((image_height / 80.0).max(1.0))
        .max(1.0)
}

fn build_spatial_line(
    line: &[&types::TextRegion],
    min_left: f32,
    char_width: f32,
) -> String {
    let mut line = line.to_vec();
    line.sort_by(|left, right| {
        left.bbox
            .left()
            .partial_cmp(&right.bbox.left())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut output = String::new();
    let mut cursor = 0usize;

    for region in line {
        let target_col = ((region.bbox.left() - min_left) / char_width)
            .round()
            .max(0.0)
            .min(240.0) as usize;
        if target_col > cursor {
            output.extend(std::iter::repeat_n(' ', target_col - cursor));
        } else if !output.is_empty() && !output.ends_with(' ') {
            output.push(' ');
        }

        let text = region.text.trim();
        output.push_str(text);
        cursor = output.chars().count();
    }

    output.trim_end().to_string()
}

fn line_center_y(line: &[&types::TextRegion]) -> f32 {
    line.iter()
        .map(|region| region.bbox.center_y())
        .sum::<f32>()
        / line.len().max(1) as f32
}

/// 版式文本入口：竖排为主时按竖排顺序，识别为表格时按列对齐，否则回退自由缩进版式。
///
/// 适配截图/网页/聊天/表格/票据等场景，让输出排版尽量贴近原图。
pub fn build_layout_text(
    regions: &[types::TextRegion],
    image_width: f32,
    image_height: f32,
) -> String {
    if is_vertical_dominant(regions) {
        return build_vertical_text(regions);
    }
    build_table_text(regions, image_width, image_height)
        .unwrap_or_else(|| build_spatial_text(regions, image_width, image_height))
}

/// 行 × 列的表格网格，空单元格为空串。
struct TableGrid {
    rows: Vec<Vec<String>>,
}

/// 尝试把检测结果还原为列对齐的表格文本。
///
/// 适用于表格、票据、发票、聊天记录、KV 等有行列结构的图片。内容不像表格时返回 `None`，
/// 由调用方回退到自由缩进的 `build_spatial_text`，避免把普通段落误判强排成网格。
pub fn build_table_text(
    regions: &[types::TextRegion],
    image_width: f32,
    image_height: f32,
) -> Option<String> {
    detect_table_grid(regions, image_width, image_height).map(|table| render_table_monospace(&table))
}

/// 尝试把表格结果导出为 CSV（逗号分隔，含标准引号转义）。内容不像表格时返回 `None`。
pub fn build_table_csv(
    regions: &[types::TextRegion],
    image_width: f32,
    image_height: f32,
) -> Option<String> {
    detect_table_grid(regions, image_width, image_height).map(|table| render_table_csv(&table))
}

fn detect_table_grid(
    regions: &[types::TextRegion],
    image_width: f32,
    image_height: f32,
) -> Option<TableGrid> {
    if regions.len() < 4 {
        return None;
    }

    let row_groups = cluster_rows_by_y(regions);
    if row_groups.len() < 2 {
        return None;
    }

    let columns = cluster_column_edges(regions, image_width, image_height);
    if columns.len() < 2 {
        return None;
    }

    // 表格判定：至少两行各自落在 ≥2 个不同列，避免把单列段落误判为表格。
    let multi_col_rows = row_groups
        .iter()
        .filter(|row| distinct_column_count(row, &columns) >= 2)
        .count();
    if multi_col_rows < 2 {
        return None;
    }

    let mut rows = Vec::with_capacity(row_groups.len());
    for row in &row_groups {
        let mut cells = vec![String::new(); columns.len()];
        for region in row {
            let col = nearest_column_index(region.bbox.left(), &columns);
            let cell = &mut cells[col];
            let text = region.text.trim();
            if cell.is_empty() {
                cell.push_str(text);
            } else {
                cell.push(' ');
                cell.push_str(text);
            }
        }
        rows.push(cells);
    }

    Some(TableGrid { rows })
}

/// 全局按 y 中心聚行：比相邻比较更鲁棒，适配乱序/表格输入。
fn cluster_rows_by_y(regions: &[types::TextRegion]) -> Vec<Vec<&types::TextRegion>> {
    let mut sorted: Vec<&types::TextRegion> = regions.iter().collect();
    sorted.sort_by(|a, b| {
        a.bbox
            .center_y()
            .partial_cmp(&b.bbox.center_y())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut rows: Vec<Vec<&types::TextRegion>> = Vec::new();
    for region in sorted {
        let placed = rows.last_mut().is_some_and(|row| {
            let row_center =
                row.iter().map(|r| r.bbox.center_y()).sum::<f32>() / row.len() as f32;
            let avg_height =
                row.iter().map(|r| r.bbox.height()).sum::<f32>() / row.len() as f32;
            let tol = avg_height.max(region.bbox.height()) * 0.6;
            if (region.bbox.center_y() - row_center).abs() <= tol {
                row.push(region);
                true
            } else {
                false
            }
        });
        if !placed {
            rows.push(vec![region]);
        }
    }

    for row in &mut rows {
        row.sort_by(|a, b| {
            a.bbox
                .left()
                .partial_cmp(&b.bbox.left())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    rows
}

/// 对所有 region 的左边缘做 1D 聚类，得到列起始位置。
fn cluster_column_edges(
    regions: &[types::TextRegion],
    image_width: f32,
    _image_height: f32,
) -> Vec<f32> {
    let mut lefts: Vec<f32> = regions.iter().map(|r| r.bbox.left()).collect();
    lefts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // 同列左边缘聚得很紧，列间距远大于字高；容差取字高量级。
    let tol = (median_region_height(regions) * 1.2)
        .max(image_width * 0.01)
        .max(4.0);

    let mut columns = Vec::new();
    let mut cluster_sum = 0.0f32;
    let mut cluster_count = 0u32;
    let mut cluster_last = f32::NEG_INFINITY;
    for x in lefts {
        if cluster_count == 0 || x - cluster_last <= tol {
            cluster_sum += x;
            cluster_count += 1;
        } else {
            columns.push(cluster_sum / cluster_count as f32);
            cluster_sum = x;
            cluster_count = 1;
        }
        cluster_last = x;
    }
    if cluster_count > 0 {
        columns.push(cluster_sum / cluster_count as f32);
    }
    columns
}

fn distinct_column_count(row: &[&types::TextRegion], columns: &[f32]) -> usize {
    let mut seen = std::collections::BTreeSet::new();
    for region in row {
        seen.insert(nearest_column_index(region.bbox.left(), columns));
    }
    seen.len()
}

fn nearest_column_index(x: f32, columns: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::MAX;
    for (index, &col) in columns.iter().enumerate() {
        let dist = (x - col).abs();
        if dist < best_dist {
            best_dist = dist;
            best = index;
        }
    }
    best
}

fn median_region_height(regions: &[types::TextRegion]) -> f32 {
    let mut heights: Vec<f32> = regions
        .iter()
        .map(|r| r.bbox.height())
        .filter(|h| *h > 0.0)
        .collect();
    if heights.is_empty() {
        return 1.0;
    }
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    heights[heights.len() / 2]
}

/// 按每列最大显示宽度对齐渲染（CJK 字符按 2 列宽计算）。
fn render_table_monospace(table: &TableGrid) -> String {
    let col_count = table.rows.iter().map(|row| row.len()).max().unwrap_or(0);
    if col_count == 0 {
        return String::new();
    }

    let mut col_width = vec![0usize; col_count];
    for row in &table.rows {
        for (index, cell) in row.iter().enumerate() {
            col_width[index] = col_width[index].max(display_width(cell));
        }
    }

    const GAP: usize = 2;
    let mut col_start = vec![0usize; col_count];
    for index in 1..col_count {
        col_start[index] = col_start[index - 1] + col_width[index - 1] + GAP;
    }

    let mut out = String::new();
    for (row_index, row) in table.rows.iter().enumerate() {
        if row_index > 0 {
            out.push('\n');
        }
        let mut cursor = 0usize;
        for (col_index, cell) in row.iter().enumerate() {
            if cell.is_empty() {
                continue;
            }
            while cursor < col_start[col_index] {
                out.push(' ');
                cursor += 1;
            }
            out.push_str(cell);
            cursor += display_width(cell);
        }
        while out.ends_with(' ') {
            out.pop();
        }
    }
    out
}

fn render_table_csv(table: &TableGrid) -> String {
    let mut out = String::new();
    for (row_index, row) in table.rows.iter().enumerate() {
        if row_index > 0 {
            out.push('\n');
        }
        for (col_index, cell) in row.iter().enumerate() {
            if col_index > 0 {
                out.push(',');
            }
            out.push_str(&csv_escape(cell));
        }
    }
    out
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// 文本显示宽度估算：CJK / 全角字符按 2 列宽，其余按 1 列宽。
fn display_width(text: &str) -> usize {
    text.chars().map(char_display_width).sum()
}

fn char_display_width(ch: char) -> usize {
    let cp = ch as u32;
    let wide = (0x1100..=0x115F).contains(&cp)      // Hangul Jamo
        || (0x2E80..=0xA4CF).contains(&cp)          // CJK 部首 .. 彝文
        || (0xAC00..=0xD7A3).contains(&cp)          // Hangul 音节
        || (0xF900..=0xFAFF).contains(&cp)          // CJK 兼容表意
        || (0xFE30..=0xFE4F).contains(&cp)          // CJK 兼容形式
        || (0xFF00..=0xFF60).contains(&cp)          // 全角 ASCII
        || (0xFFE0..=0xFFE6).contains(&cp)          // 全角符号
        || (0x1F300..=0x1FAFF).contains(&cp)        // emoji
        || (0x20000..=0x3FFFD).contains(&cp); // CJK 扩展
    if wide { 2 } else { 1 }
}

/// 整图是否以竖排文本为主（默认横排，仅在多数 region 为竖排时启用竖排顺序）。
fn is_vertical_dominant(regions: &[types::TextRegion]) -> bool {
    if regions.len() < 2 {
        return false;
    }
    let vertical = regions
        .iter()
        .filter(|r| r.direction == types::TextDirection::Vertical)
        .count();
    vertical * 2 > regions.len()
}

/// 竖排阅读顺序：列从右到左，列内从上到下；每列输出为一行。
fn build_vertical_text(regions: &[types::TextRegion]) -> String {
    let mut columns: Vec<Vec<&types::TextRegion>> = Vec::new();
    let mut sorted: Vec<&types::TextRegion> = regions.iter().collect();
    // 先按 center_x 降序（右列优先）。
    sorted.sort_by(|a, b| {
        b.bbox
            .center_x()
            .partial_cmp(&a.bbox.center_x())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for region in sorted {
        let placed = columns.last_mut().is_some_and(|col| {
            let col_center =
                col.iter().map(|r| r.bbox.center_x()).sum::<f32>() / col.len() as f32;
            let avg_width = col.iter().map(|r| r.bbox.width()).sum::<f32>() / col.len() as f32;
            let tol = avg_width.max(region.bbox.width()) * 0.6;
            if (region.bbox.center_x() - col_center).abs() <= tol {
                col.push(region);
                true
            } else {
                false
            }
        });
        if !placed {
            columns.push(vec![region]);
        }
    }

    let mut out = String::new();
    for (col_index, mut col) in columns.into_iter().enumerate() {
        col.sort_by(|a, b| {
            a.bbox
                .top()
                .partial_cmp(&b.bbox.top())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if col_index > 0 {
            out.push('\n');
        }
        for region in col {
            out.push_str(region.text.trim());
        }
    }
    out
}

#[derive(Debug, Clone)]
struct ReconstructedLine {
    center_y: f32,
    height: f32,
    text: String,
    line_number_candidate: Option<u32>,
}

fn reconstruct_layout_lines(grouped_lines: &[Vec<&types::TextRegion>]) -> Vec<ReconstructedLine> {
    let lines: Vec<ReconstructedLine> = grouped_lines
        .iter()
        .map(|line| {
            let text = join_line_regions(line);
            let center_y = line
                .iter()
                .map(|region| region.bbox.center_y())
                .sum::<f32>()
                / line.len() as f32;
            let height =
                line.iter().map(|region| region.bbox.height()).sum::<f32>() / line.len() as f32;

            ReconstructedLine {
                center_y,
                height,
                line_number_candidate: extract_leading_number_candidate(&text),
                text,
            }
        })
        .collect();

    if lines.len() <= 1 {
        return lines;
    }

    let line_pitch = estimate_line_pitch(&lines);
    let mut expanded = Vec::with_capacity(lines.len());

    for (index, line) in lines.iter().enumerate() {
        expanded.push(line.clone());

        let Some(next_line) = lines.get(index + 1) else {
            continue;
        };

        let delta_y = (next_line.center_y - line.center_y).max(0.0);
        let slot_span = ((delta_y / line_pitch).round() as isize).max(1) as usize;
        let missing = slot_span.saturating_sub(1).min(32);

        for gap_index in 0..missing {
            expanded.push(ReconstructedLine {
                center_y: line.center_y + line_pitch * (gap_index + 1) as f32,
                height: line_pitch,
                text: String::new(),
                line_number_candidate: None,
            });
        }
    }

    expanded
}

fn estimate_line_pitch(lines: &[ReconstructedLine]) -> f32 {
    let mut deltas: Vec<f32> = lines
        .windows(2)
        .filter_map(|pair| {
            let delta = pair[1].center_y - pair[0].center_y;
            (delta.is_finite() && delta > 0.0).then_some(delta)
        })
        .collect();
    deltas.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

    if !deltas.is_empty() {
        let sample_len = deltas.len().div_ceil(2);
        let sample = &deltas[..sample_len];
        let middle = sample.len() / 2;
        return if sample.len() % 2 == 0 {
            (sample[middle - 1] + sample[middle]) * 0.5
        } else {
            sample[middle]
        }
        .max(1.0);
    }

    (lines.iter().map(|line| line.height).sum::<f32>() / lines.len().max(1) as f32).max(1.0)
}

fn detect_line_numbering_start(
    lines: &[ReconstructedLine],
    original_line_count: usize,
) -> Option<u32> {
    let candidates: Vec<(usize, u32)> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            line.line_number_candidate
                .map(|candidate| (index, candidate))
        })
        .collect();

    if candidates.len() < 3 {
        return None;
    }

    let has_blank_slots = lines.len() > original_line_count;
    let max_candidate = candidates
        .iter()
        .map(|(_, candidate)| *candidate)
        .max()
        .unwrap_or(0);
    if !has_blank_slots && max_candidate as usize <= original_line_count + 2 {
        return None;
    }

    let mut best: Option<(u32, usize, usize)> = None;
    for &(line_index, candidate) in &candidates {
        let Some(start) = candidate.checked_sub(line_index as u32) else {
            continue;
        };

        let exact_matches = candidates
            .iter()
            .filter(|&&(index, value)| start + index as u32 == value)
            .count();
        let near_matches = candidates
            .iter()
            .filter(|&&(index, value)| {
                (start as i64 + index as i64 - value as i64).unsigned_abs() <= 1
            })
            .count();

        if exact_matches < 3 {
            continue;
        }

        match best {
            Some((_, best_exact, best_near))
                if exact_matches < best_exact
                    || (exact_matches == best_exact && near_matches <= best_near) => {}
            _ => best = Some((start, exact_matches, near_matches)),
        }
    }

    best.map(|(start, _, _)| start)
}

fn extract_leading_number_candidate(text: &str) -> Option<u32> {
    let digit_count = text.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digit_count == 0 {
        return None;
    }

    text.get(..digit_count)?.parse::<u32>().ok()
}

fn strip_matching_line_number_prefix(text: &str, line_number: u32) -> String {
    let prefix = line_number.to_string();
    if let Some(rest) = text.strip_prefix(&prefix) {
        return rest.trim_start().to_string();
    }

    text.to_string()
}

fn push_right_aligned_number(target: &mut String, number: u32, width: usize) {
    let digits = number.to_string();
    if width > digits.len() {
        target.extend(std::iter::repeat_n(' ', width - digits.len()));
    }
    target.push_str(&digits);
}

fn prepare_detections_for_recognition(
    detections: &[(BoundingBox, f32)],
    image_width: f32,
) -> Vec<(BoundingBox, f32)> {
    if detections.is_empty() {
        return Vec::new();
    }

    let gutter_right = detect_left_gutter(detections, image_width);
    let mut prepared: Vec<(BoundingBox, f32)> = Vec::with_capacity(detections.len());

    for (bbox, confidence) in detections.iter().copied() {
        if let Some(gutter_right) = gutter_right
            && is_likely_gutter_box(&bbox, image_width, gutter_right)
        {
            continue;
        }

        if let Some((last_bbox, last_confidence)) = prepared.last_mut()
            && should_merge_detections(last_bbox, &bbox)
        {
            *last_bbox = last_bbox.union(&bbox);
            *last_confidence = last_confidence.max(confidence);
            continue;
        }

        prepared.push((bbox, confidence));
    }

    prepared
}

fn detect_left_gutter(detections: &[(BoundingBox, f32)], image_width: f32) -> Option<f32> {
    let mut gutter_candidates = Vec::new();
    for (bbox, _) in detections {
        if !is_likely_gutter_box(bbox, image_width, image_width * 0.12) {
            continue;
        }

        let has_text_on_same_row = detections.iter().any(|(other, _)| {
            !std::ptr::eq(other, bbox)
                && shares_text_line(bbox, other)
                && other.left() > bbox.right() + bbox.height() * 0.8
                && other.width() > bbox.width() * 1.5
        });
        if has_text_on_same_row {
            gutter_candidates.push(*bbox);
        }
    }

    if gutter_candidates.len() < 5 {
        return None;
    }

    let max_right = gutter_candidates
        .iter()
        .map(BoundingBox::right)
        .fold(0.0f32, f32::max);
    let avg_height = gutter_candidates
        .iter()
        .map(BoundingBox::height)
        .sum::<f32>()
        / gutter_candidates.len() as f32;
    Some((max_right + avg_height * 0.8).min(image_width * 0.15))
}

fn is_likely_gutter_box(bbox: &BoundingBox, image_width: f32, gutter_right: f32) -> bool {
    let width_limit = (image_width * 0.08).max(36.0);
    let height_limit = (image_width * 0.05).max(24.0);
    bbox.left() <= gutter_right
        && bbox.right() <= gutter_right
        && bbox.width() <= width_limit
        && bbox.height() <= height_limit
        && bbox.width() <= bbox.height() * 2.2 + 8.0
}

fn should_merge_detections(left: &BoundingBox, right: &BoundingBox) -> bool {
    // 旋转框合并会经 union 退化为 AABB 丢失旋转信息，故仅合并近水平的轴对齐框。
    if !left.is_axis_aligned() || !right.is_axis_aligned() {
        return false;
    }
    if !shares_text_line(left, right) {
        return false;
    }

    let avg_height = (left.height() + right.height()) * 0.5;
    let gap = right.left() - left.right();
    let overlap = left.right().min(right.right()) - left.left().max(right.left());
    overlap >= avg_height * 0.4 || gap <= avg_height * 1.2
}

fn shares_text_line(left: &BoundingBox, right: &BoundingBox) -> bool {
    let center_delta = (left.center_y() - right.center_y()).abs();
    let avg_height = (left.height() + right.height()) * 0.5;
    let vertical_overlap = left.bottom().min(right.bottom()) - left.top().max(right.top());
    center_delta <= avg_height * 0.7 || vertical_overlap >= avg_height * 0.5
}

fn group_regions_by_line(regions: &[types::TextRegion]) -> Vec<Vec<&types::TextRegion>> {
    let mut lines: Vec<Vec<&types::TextRegion>> = Vec::new();

    for region in regions {
        if let Some(current_line) = lines.last_mut()
            && current_line
                .last()
                .is_some_and(|last| shares_text_line(&last.bbox, &region.bbox))
        {
            current_line.push(region);
            continue;
        }

        lines.push(vec![region]);
    }

    lines
}

fn join_line_regions(line: &[&types::TextRegion]) -> String {
    let mut text = String::with_capacity(
        line.iter().map(|region| region.text.len()).sum::<usize>() + line.len().saturating_sub(1),
    );

    for (index, region) in line.iter().enumerate() {
        if index > 0 {
            let previous = line[index - 1];
            let gap = region.bbox.left() - previous.bbox.right();
            let avg_height = (region.bbox.height() + previous.bbox.height()) * 0.5;
            if gap > avg_height * 0.7
                && needs_space_between(&previous.text, &region.text)
                && !text.ends_with(' ')
            {
                text.push(' ');
            }
        }
        text.push_str(region.text.trim());
    }

    text
}

fn needs_space_between(left: &str, right: &str) -> bool {
    let left_char = left.chars().next_back();
    let right_char = right.chars().next();
    match (left_char, right_char) {
        (Some(left), Some(right)) => left.is_ascii_alphanumeric() && right.is_ascii_alphanumeric(),
        _ => false,
    }
}

pub(crate) fn log_ocr_stage_timings(
    detector_name: &str,
    recognizer_name: &str,
    region_count: usize,
    timings: &OcrStageTimings,
) {
    tracing::debug!(
        detector = detector_name,
        recognizer = recognizer_name,
        regions = region_count,
        detect_ms = timings.detect.as_millis() as u64,
        crop_ms = timings.crop.as_millis() as u64,
        classify_ms = timings.classify.as_millis() as u64,
        recognize_ms = timings.recognize.as_millis() as u64,
        join_ms = timings.join.as_millis() as u64,
        "OCR 分阶段耗时"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BoundingBox, TextDirection, TextRegion};

    #[test]
    fn test_from_config_prefers_models_dir_onnx_smoke() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let models_dir =
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
                let mut config = OcrConfig::default();
                config.weights_dir = models_dir.clone();
                config.dict_path = models_dir.join("ppocr_keys_v1.txt");

                let engine = OcrEngine::from_config(&config).unwrap();
                let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                    128,
                    64,
                    image::Rgb([255, 255, 255]),
                ));
                let result = engine.run(&image).unwrap();

                assert!(result.full_text.is_empty());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn test_prepare_detections_filters_gutter_and_merges_same_line() {
        let detections = vec![
            (BoundingBox::from_rect(10.0, 10.0, 16.0, 12.0), 0.9),
            (BoundingBox::from_rect(48.0, 10.0, 120.0, 12.0), 0.9),
            (BoundingBox::from_rect(180.0, 10.0, 90.0, 12.0), 0.8),
            (BoundingBox::from_rect(11.0, 30.0, 16.0, 12.0), 0.9),
            (BoundingBox::from_rect(48.0, 30.0, 120.0, 12.0), 0.9),
            (BoundingBox::from_rect(11.0, 50.0, 16.0, 12.0), 0.9),
            (BoundingBox::from_rect(48.0, 50.0, 120.0, 12.0), 0.9),
            (BoundingBox::from_rect(11.0, 70.0, 16.0, 12.0), 0.9),
            (BoundingBox::from_rect(48.0, 70.0, 120.0, 12.0), 0.9),
            (BoundingBox::from_rect(11.0, 90.0, 16.0, 12.0), 0.9),
            (BoundingBox::from_rect(48.0, 90.0, 120.0, 12.0), 0.9),
        ];

        let prepared = prepare_detections_for_recognition(&detections, 400.0);

        assert_eq!(prepared.len(), 5);
        assert!(prepared.iter().all(|(bbox, _)| bbox.left() >= 48.0));
        assert!(prepared[0].0.right() >= 270.0);
    }

    #[test]
    fn test_build_full_text_groups_same_line_regions() {
        let regions = vec![
            TextRegion {
                bbox: BoundingBox::from_rect(50.0, 10.0, 80.0, 12.0),
                confidence: 0.9,
                text: "hello".to_string(),
                direction: TextDirection::Horizontal,
            },
            TextRegion {
                bbox: BoundingBox::from_rect(140.0, 10.0, 80.0, 12.0),
                confidence: 0.9,
                text: "world".to_string(),
                direction: TextDirection::Horizontal,
            },
            TextRegion {
                bbox: BoundingBox::from_rect(50.0, 34.0, 80.0, 12.0),
                confidence: 0.9,
                text: "next".to_string(),
                direction: TextDirection::Horizontal,
            },
        ];

        assert_eq!(build_full_text(&regions), "hello world\nnext");
    }

    #[test]
    fn test_build_full_text_preserves_blank_rows_and_line_numbers() {
        let regions = vec![
            TextRegion {
                bbox: BoundingBox::from_rect(40.0, 10.0, 120.0, 12.0),
                confidence: 0.9,
                text: "1# AIOCR".to_string(),
                direction: TextDirection::Horizontal,
            },
            TextRegion {
                bbox: BoundingBox::from_rect(40.0, 50.0, 220.0, 12.0),
                confidence: 0.9,
                text: "3纯 Rust OCR".to_string(),
                direction: TextDirection::Horizontal,
            },
            TextRegion {
                bbox: BoundingBox::from_rect(40.0, 70.0, 180.0, 12.0),
                confidence: 0.9,
                text: "4- item".to_string(),
                direction: TextDirection::Horizontal,
            },
        ];

        assert_eq!(
            build_full_text(&regions),
            "1  # AIOCR\n2\n3  纯 Rust OCR\n4  - item"
        );
    }

    #[test]
    fn test_build_spatial_text_preserves_columns_and_large_vertical_gaps() {
        let regions = vec![
            TextRegion {
                bbox: BoundingBox::from_rect(20.0, 10.0, 40.0, 12.0),
                confidence: 0.9,
                text: "姓名".to_string(),
                direction: TextDirection::Horizontal,
            },
            TextRegion {
                bbox: BoundingBox::from_rect(180.0, 10.0, 60.0, 12.0),
                confidence: 0.9,
                text: "金额".to_string(),
                direction: TextDirection::Horizontal,
            },
            TextRegion {
                bbox: BoundingBox::from_rect(20.0, 50.0, 60.0, 12.0),
                confidence: 0.9,
                text: "张三".to_string(),
                direction: TextDirection::Horizontal,
            },
            TextRegion {
                bbox: BoundingBox::from_rect(180.0, 50.0, 80.0, 12.0),
                confidence: 0.9,
                text: "128.00".to_string(),
                direction: TextDirection::Horizontal,
            },
            TextRegion {
                bbox: BoundingBox::from_rect(20.0, 130.0, 80.0, 12.0),
                confidence: 0.9,
                text: "备注".to_string(),
                direction: TextDirection::Horizontal,
            },
        ];

        let text = build_spatial_text(&regions, 320.0, 180.0);

        assert!(text.contains("姓名"));
        assert!(text.contains("金额"));
        assert!(text.contains("张三"));
        assert!(text.contains("128.00"));
        assert!(text.contains("\n\n备注"));
        let first_line = text.lines().next().unwrap_or_default();
        assert!(first_line.find("金额").unwrap() > first_line.find("姓名").unwrap());
    }

    #[test]
    fn test_detect_line_numbering_start_ignores_outlier_candidate() {
        let lines = vec![
            ReconstructedLine {
                center_y: 0.0,
                height: 12.0,
                text: "1 title".to_string(),
                line_number_candidate: Some(1),
            },
            ReconstructedLine {
                center_y: 20.0,
                height: 12.0,
                text: String::new(),
                line_number_candidate: None,
            },
            ReconstructedLine {
                center_y: 40.0,
                height: 12.0,
                text: "3 body".to_string(),
                line_number_candidate: Some(3),
            },
            ReconstructedLine {
                center_y: 60.0,
                height: 12.0,
                text: "4 body".to_string(),
                line_number_candidate: Some(4),
            },
            ReconstructedLine {
                center_y: 80.0,
                height: 12.0,
                text: "241 body".to_string(),
                line_number_candidate: Some(241),
            },
        ];

        assert_eq!(detect_line_numbering_start(&lines, 4), Some(1));
    }

    fn region(text: &str, x: f32, y: f32, w: f32, h: f32, dir: TextDirection) -> TextRegion {
        TextRegion {
            bbox: BoundingBox::from_rect(x, y, w, h),
            confidence: 0.9,
            text: text.to_string(),
            direction: dir,
        }
    }

    fn table_regions() -> Vec<TextRegion> {
        vec![
            region("姓名", 20.0, 10.0, 40.0, 12.0, TextDirection::Horizontal),
            region("金额", 180.0, 10.0, 60.0, 12.0, TextDirection::Horizontal),
            region("张三", 20.0, 40.0, 40.0, 12.0, TextDirection::Horizontal),
            region("100", 180.0, 40.0, 40.0, 12.0, TextDirection::Horizontal),
            region("李四", 22.0, 70.0, 40.0, 12.0, TextDirection::Horizontal),
            region("2000", 182.0, 70.0, 50.0, 12.0, TextDirection::Horizontal),
        ]
    }

    #[test]
    fn test_build_table_text_aligns_columns() {
        let regions = table_regions();
        let text = build_table_text(&regions, 260.0, 100.0).expect("应识别为表格");

        assert_eq!(text, "姓名  金额\n张三  100\n李四  2000");
    }

    #[test]
    fn test_build_table_csv_emits_rows() {
        let regions = table_regions();
        let csv = build_table_csv(&regions, 260.0, 100.0).expect("应识别为表格");

        assert_eq!(csv, "姓名,金额\n张三,100\n李四,2000");
    }

    #[test]
    fn test_build_table_text_returns_none_for_single_column_prose() {
        let regions = vec![
            region("第一行普通段落", 20.0, 10.0, 200.0, 12.0, TextDirection::Horizontal),
            region("第二行普通段落", 20.0, 34.0, 210.0, 12.0, TextDirection::Horizontal),
            region("第三行普通段落", 20.0, 58.0, 180.0, 12.0, TextDirection::Horizontal),
            region("第四行普通段落", 20.0, 82.0, 220.0, 12.0, TextDirection::Horizontal),
        ];

        assert!(build_table_text(&regions, 260.0, 120.0).is_none());
    }

    #[test]
    fn test_build_layout_text_uses_vertical_order_when_dominant() {
        // 两列竖排：右列 x≈106 先读，列内从上到下；左列 x≈26 后读。
        let regions = vec![
            region("上", 100.0, 10.0, 12.0, 24.0, TextDirection::Vertical),
            region("下", 100.0, 40.0, 12.0, 24.0, TextDirection::Vertical),
            region("左", 20.0, 10.0, 12.0, 24.0, TextDirection::Vertical),
            region("右", 20.0, 40.0, 12.0, 24.0, TextDirection::Vertical),
        ];

        let text = build_layout_text(&regions, 140.0, 80.0);

        assert_eq!(text, "上下\n左右");
    }
}
