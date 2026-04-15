use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use aiocr_core::Recognizer;
use aiocr_core::error::OcrError;
use aiocr_core::preprocess::{fit_to_canvas_grayscale, grayscale_to_ink};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::error::TrainError;

pub const TEMPLATE_MODEL_FILE: &str = "template-model.json";
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.40;

/// 模板模型中的单个文本条目。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateEntry {
    pub label: String,
    pub sample_count: usize,
    pub feature: Vec<f32>,
}

/// 模板模型指标。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemplateMetrics {
    pub train_samples: usize,
    pub validation_samples: usize,
    pub accuracy: f32,
    pub avg_loss: f32,
}

/// 本地模板识别模型。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateModel {
    pub name: String,
    pub created_at: u64,
    pub input_width: u32,
    pub input_height: u32,
    pub similarity_threshold: f32,
    pub templates: Vec<TemplateEntry>,
    pub metrics: TemplateMetrics,
}

/// 供 GUI 展示的模型摘要。
#[derive(Debug, Clone)]
pub struct TemplateModelInfo {
    pub model_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub name: String,
    pub created_at: u64,
    pub template_count: usize,
    pub input_width: u32,
    pub input_height: u32,
    pub accuracy: f32,
    pub avg_loss: f32,
}

/// 本地模板识别器。
#[derive(Debug, Clone)]
pub struct TemplateRecognizer {
    model: TemplateModel,
}

impl TemplateModel {
    /// 将模型保存到产物根目录下的新子目录。
    pub fn save_to_root(&self, root: &Path) -> Result<PathBuf, TrainError> {
        std::fs::create_dir_all(root)?;

        let dir_name = format!("{}-{}", self.created_at, sanitize_name(&self.name));
        let model_dir = root.join(dir_name);
        std::fs::create_dir_all(&model_dir)?;
        self.save_to_dir(&model_dir)?;

        Ok(model_dir)
    }

    /// 将模型保存到指定目录。
    pub fn save_to_dir(&self, dir: &Path) -> Result<(), TrainError> {
        std::fs::create_dir_all(dir)?;
        let content = serde_json::to_vec_pretty(self)
            .map_err(|e| TrainError::Export(format!("序列化模板模型失败: {e}")))?;
        std::fs::write(dir.join(TEMPLATE_MODEL_FILE), content)?;
        Ok(())
    }

    /// 从模型目录加载。
    pub fn load_from_dir(dir: &Path) -> Result<Self, TrainError> {
        Self::load_from_file(&dir.join(TEMPLATE_MODEL_FILE))
    }

    /// 从模型文件加载。
    pub fn load_from_file(path: &Path) -> Result<Self, TrainError> {
        let content = std::fs::read(path)
            .map_err(|e| TrainError::Dataset(format!("读取模型文件 {path:?} 失败: {e}")))?;
        serde_json::from_slice(&content)
            .map_err(|e| TrainError::Dataset(format!("解析模板模型 {path:?} 失败: {e}")))
    }

    /// 将聚合后的特征构造成模板模型。
    pub fn from_grouped_features(
        name: String,
        input_width: u32,
        input_height: u32,
        metrics: TemplateMetrics,
        grouped: HashMap<String, Vec<Vec<f32>>>,
    ) -> Result<Self, TrainError> {
        let mut templates = Vec::with_capacity(grouped.len());
        for (label, samples) in grouped {
            let feature = average_features(&samples)?;
            templates.push(TemplateEntry {
                label,
                sample_count: samples.len(),
                feature,
            });
        }

        templates.sort_by(|a, b| a.label.cmp(&b.label));

        Ok(Self {
            name,
            created_at: now_unix_secs(),
            input_width,
            input_height,
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            templates,
            metrics,
        })
    }
}

impl TemplateModelInfo {
    pub fn from_model_dir(model_dir: PathBuf, model: &TemplateModel) -> Self {
        Self {
            manifest_path: model_dir.join(TEMPLATE_MODEL_FILE),
            model_dir,
            name: model.name.clone(),
            created_at: model.created_at,
            template_count: model.templates.len(),
            input_width: model.input_width,
            input_height: model.input_height,
            accuracy: model.metrics.accuracy,
            avg_loss: model.metrics.avg_loss,
        }
    }

    pub fn display_name(&self) -> String {
        format!("{} ({:.0}%)", self.name, self.accuracy * 100.0)
    }
}

impl TemplateRecognizer {
    pub fn from_model(model: TemplateModel) -> Self {
        Self { model }
    }

    pub fn load_from_dir(dir: &Path) -> Result<Self, TrainError> {
        Ok(Self::from_model(TemplateModel::load_from_dir(dir)?))
    }

    pub fn model(&self) -> &TemplateModel {
        &self.model
    }
}

impl Recognizer for TemplateRecognizer {
    fn recognize(&self, crop: &DynamicImage) -> Result<(String, f32), OcrError> {
        if self.model.templates.is_empty() {
            return Err(OcrError::Inference(format!(
                "模板模型 {} 不包含任何模板条目",
                self.model.name
            )));
        }

        let feature = extract_feature(crop, self.model.input_width, self.model.input_height);

        let best = self
            .model
            .templates
            .iter()
            .map(|template| {
                let similarity = cosine_similarity(&feature, &template.feature);
                (template, similarity)
            })
            .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal));

        let Some((template, similarity)) = best else {
            return Ok((String::new(), 0.0));
        };

        let confidence = similarity.clamp(0.0, 1.0);
        if confidence < self.model.similarity_threshold {
            return Ok((String::new(), confidence));
        }

        Ok((template.label.clone(), confidence))
    }

    fn name(&self) -> &str {
        &self.model.name
    }
}

/// 提取用于模板匹配的图像特征。
pub fn extract_feature(image: &DynamicImage, input_width: u32, input_height: u32) -> Vec<f32> {
    let canvas = fit_to_canvas_grayscale(image, input_width, input_height);
    let ink = grayscale_to_ink(&canvas);
    normalize_feature(ink)
}

/// 列出产物目录中的全部模板模型。
pub fn list_models(root: &Path) -> Result<Vec<TemplateModelInfo>, TrainError> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        let model_dir = entry.path();
        let manifest = model_dir.join(TEMPLATE_MODEL_FILE);
        if !manifest.exists() {
            continue;
        }

        match TemplateModel::load_from_dir(&model_dir) {
            Ok(model) => models.push(TemplateModelInfo::from_model_dir(model_dir, &model)),
            Err(err) => tracing::warn!("跳过无法加载的模板模型 {:?}: {err}", model_dir),
        }
    }

    models.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    Ok(models)
}

fn average_features(samples: &[Vec<f32>]) -> Result<Vec<f32>, TrainError> {
    let Some(first) = samples.first() else {
        return Err(TrainError::Model("模板样本为空".to_string()));
    };

    let mut sum = vec![0.0f32; first.len()];
    for sample in samples {
        if sample.len() != sum.len() {
            return Err(TrainError::Model("模板特征维度不一致".to_string()));
        }
        for (slot, value) in sum.iter_mut().zip(sample) {
            *slot += *value;
        }
    }

    let count = samples.len() as f32;
    for value in &mut sum {
        *value /= count;
    }

    Ok(normalize_feature(sum))
}

fn normalize_feature(mut feature: Vec<f32>) -> Vec<f32> {
    let norm = feature
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    for value in &mut feature {
        *value /= norm;
    }
    feature
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return 0.0;
    }

    left.iter()
        .zip(right)
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>()
        .clamp(0.0, 1.0)
}

fn sanitize_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect();

    sanitized
        .trim_matches('-')
        .to_string()
        .if_empty("template-model")
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

trait IfEmpty {
    fn if_empty(self, fallback: &str) -> String;
}

impl IfEmpty for String {
    fn if_empty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_bounds() {
        let left = vec![1.0, 0.0, 0.0];
        let right = vec![1.0, 0.0, 0.0];
        let other = vec![0.0, 1.0, 0.0];

        assert_eq!(cosine_similarity(&left, &right), 1.0);
        assert_eq!(cosine_similarity(&left, &other), 0.0);
    }
}
