use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::TrainError;

pub const AI_MODEL_MANIFEST_FILE: &str = "ai-recognizer.json";
pub const AI_MODEL_WEIGHTS_FILE: &str = "rec.bpk";
pub const AI_MODEL_DICT_FILE: &str = "dict.txt";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AiModelMetrics {
    pub train_samples: usize,
    pub validation_samples: usize,
    pub accuracy: f32,
    pub avg_loss: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiModelManifest {
    pub name: String,
    pub created_at: u64,
    pub base_model_name: String,
    pub weights_file: String,
    pub dict_file: String,
    pub metrics: AiModelMetrics,
}

#[derive(Debug, Clone)]
pub struct AiModelInfo {
    pub model_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub weights_path: PathBuf,
    pub dict_path: PathBuf,
    pub name: String,
    pub created_at: u64,
    pub base_model_name: String,
    pub train_samples: usize,
    pub validation_samples: usize,
    pub accuracy: f32,
    pub avg_loss: f32,
}

impl AiModelManifest {
    pub fn new(name: String, base_model_name: String, metrics: AiModelMetrics) -> Self {
        Self {
            name,
            created_at: now_unix_secs(),
            base_model_name,
            weights_file: AI_MODEL_WEIGHTS_FILE.to_string(),
            dict_file: AI_MODEL_DICT_FILE.to_string(),
            metrics,
        }
    }

    pub fn save_to_root(
        &self,
        root: &Path,
        weights: &[u8],
        dict_source: &Path,
    ) -> Result<PathBuf, TrainError> {
        std::fs::create_dir_all(root)?;

        let dir_name = format!("{}-{}", self.created_at, sanitize_name(&self.name));
        let model_dir = root.join(dir_name);
        self.save_to_dir(&model_dir, weights, dict_source)?;
        Ok(model_dir)
    }

    pub fn save_to_dir(
        &self,
        dir: &Path,
        weights: &[u8],
        dict_source: &Path,
    ) -> Result<(), TrainError> {
        std::fs::create_dir_all(dir)?;

        std::fs::write(dir.join(&self.weights_file), weights).map_err(|err| {
            TrainError::Export(format!(
                "写入识别权重 {} 失败: {err}",
                dir.join(&self.weights_file).display()
            ))
        })?;

        std::fs::copy(dict_source, dir.join(&self.dict_file)).map_err(|err| {
            TrainError::Export(format!("复制字典 {} 失败: {err}", dict_source.display()))
        })?;

        let content = serde_json::to_vec_pretty(self)
            .map_err(|err| TrainError::Export(format!("序列化 AI 模型清单失败: {err}")))?;
        std::fs::write(dir.join(AI_MODEL_MANIFEST_FILE), content).map_err(|err| {
            TrainError::Export(format!(
                "写入模型清单 {} 失败: {err}",
                dir.join(AI_MODEL_MANIFEST_FILE).display()
            ))
        })?;

        Ok(())
    }

    pub fn load_from_dir(dir: &Path) -> Result<Self, TrainError> {
        let manifest_path = dir.join(AI_MODEL_MANIFEST_FILE);
        let content = std::fs::read(&manifest_path).map_err(|err| {
            TrainError::Dataset(format!(
                "读取模型清单 {} 失败: {err}",
                manifest_path.display()
            ))
        })?;
        serde_json::from_slice(&content).map_err(|err| {
            TrainError::Dataset(format!(
                "解析模型清单 {} 失败: {err}",
                manifest_path.display()
            ))
        })
    }

    pub fn to_info(&self, model_dir: PathBuf) -> AiModelInfo {
        AiModelInfo {
            manifest_path: model_dir.join(AI_MODEL_MANIFEST_FILE),
            weights_path: model_dir.join(&self.weights_file),
            dict_path: model_dir.join(&self.dict_file),
            model_dir,
            name: self.name.clone(),
            created_at: self.created_at,
            base_model_name: self.base_model_name.clone(),
            train_samples: self.metrics.train_samples,
            validation_samples: self.metrics.validation_samples,
            accuracy: self.metrics.accuracy,
            avg_loss: self.metrics.avg_loss,
        }
    }
}

impl AiModelInfo {
    pub fn load_from_dir(dir: &Path) -> Result<Self, TrainError> {
        let manifest = AiModelManifest::load_from_dir(dir)?;
        Ok(manifest.to_info(dir.to_path_buf()))
    }

    pub fn display_name(&self) -> String {
        format!("{} ({:.0}%)", self.name, self.accuracy * 100.0)
    }
}

pub fn list_ai_models(root: &Path) -> Result<Vec<AiModelInfo>, TrainError> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        let dir = entry.path();
        let manifest_path = dir.join(AI_MODEL_MANIFEST_FILE);
        if !manifest_path.exists() {
            continue;
        }

        match AiModelInfo::load_from_dir(&dir) {
            Ok(model) => models.push(model),
            Err(err) => tracing::warn!("跳过无法加载的 AI 模型 {}: {err}", dir.display()),
        }
    }

    models.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    Ok(models)
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
        .if_empty("ai-recognizer")
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
