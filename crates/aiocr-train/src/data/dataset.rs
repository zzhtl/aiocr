use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::data::dataset::Dataset;

use crate::error::TrainError;

/// OCR 数据集样本
#[derive(Debug, Clone)]
pub struct OcrSample {
    pub image_path: PathBuf,
    pub label: String,
}

/// OCR 数据集项（处理后）
#[derive(Debug, Clone)]
pub struct OcrItem {
    pub image_path: PathBuf,
    pub label_indices: Vec<usize>,
    pub label_text: String,
}

/// OCR 数据集
///
/// 数据格式：
/// ```text
/// dataset/
/// ├── images/
/// │   ├── 0001.jpg
/// │   └── ...
/// └── labels.txt    # "0001.jpg\t识别文本\n..."
/// ```
pub struct OcrDataset {
    items: Vec<OcrItem>,
}

impl OcrDataset {
    /// 从目录加载数据集
    pub fn from_dir(dir: &Path, char_to_idx: &HashMap<char, usize>) -> Result<Self, TrainError> {
        let labels_path = dir.join("labels.txt");
        let content = std::fs::read_to_string(&labels_path)
            .map_err(|e| TrainError::Dataset(format!("{labels_path:?}: {e}")))?;

        let images_dir = dir.join("images");
        let mut items = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, '\t').collect();
            if parts.len() != 2 {
                tracing::warn!("跳过格式错误的行: {line}");
                continue;
            }

            let image_path = images_dir.join(parts[0]);
            let label_text = parts[1].to_string();

            // 将标签文本转换为字符索引
            let label_indices: Vec<usize> = label_text
                .chars()
                .filter_map(|c| char_to_idx.get(&c).copied())
                .collect();

            if label_indices.is_empty() {
                tracing::warn!("跳过无法编码的标签: {label_text}");
                continue;
            }

            items.push(OcrItem {
                image_path,
                label_indices,
                label_text,
            });
        }

        tracing::info!("加载数据集: {} 个样本", items.len());

        Ok(Self { items })
    }

    /// 加载仅用于模板训练的数据集，不要求字符级编码。
    pub fn from_dir_raw(dir: &Path) -> Result<Self, TrainError> {
        let labels_path = dir.join("labels.txt");
        let content = std::fs::read_to_string(&labels_path)
            .map_err(|e| TrainError::Dataset(format!("{labels_path:?}: {e}")))?;

        let images_dir = dir.join("images");
        let mut items = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, '\t').collect();
            if parts.len() != 2 {
                tracing::warn!("跳过格式错误的行: {line}");
                continue;
            }

            let image_path = images_dir.join(parts[0]);
            let label_text = parts[1].to_string();
            if label_text.trim().is_empty() {
                continue;
            }

            items.push(OcrItem {
                image_path,
                label_indices: Vec::new(),
                label_text,
            });
        }

        tracing::info!("加载原始数据集: {} 个样本", items.len());
        Ok(Self { items })
    }

    /// 获取全部样本。
    pub fn items(&self) -> &[OcrItem] {
        &self.items
    }

    /// 从字典文件构建字符到索引的映射
    pub fn build_char_map(dict_path: &Path) -> Result<HashMap<char, usize>, TrainError> {
        let content = std::fs::read_to_string(dict_path)
            .map_err(|e| TrainError::Dataset(format!("{dict_path:?}: {e}")))?;

        let entries: Vec<&str> = content
            .lines()
            .map(|line| line.strip_suffix('\r').unwrap_or(line))
            .filter(|line| !line.is_empty())
            .collect();
        let has_ascii_space = entries.iter().any(|entry| *entry == " ");

        let mut char_to_idx = HashMap::new();
        // index 0 = blank
        for (i, entry) in entries.iter().enumerate() {
            let normalized = if !has_ascii_space && *entry == "\u{3000}" {
                " "
            } else {
                entry
            };
            if let Some(c) = normalized.chars().next() {
                char_to_idx.insert(c, i + 1); // +1 因为 0 是 blank
            }
        }

        Ok(char_to_idx)
    }
}

impl Dataset<OcrItem> for OcrDataset {
    fn get(&self, index: usize) -> Option<OcrItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::OcrDataset;

    #[test]
    fn test_build_char_map_preserves_ascii_space_entry() {
        let temp_root = std::env::temp_dir().join(format!(
            "aiocr-dataset-test-space-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("main")
        ));
        std::fs::create_dir_all(&temp_root).unwrap();
        let dict_path = temp_root.join("dict.txt");
        std::fs::write(&dict_path, "a\n \nb").unwrap();

        let char_map = OcrDataset::build_char_map(&dict_path).unwrap();

        assert_eq!(char_map.get(&'a'), Some(&1));
        assert_eq!(char_map.get(&' '), Some(&2));
        assert_eq!(char_map.get(&'b'), Some(&3));

        let _ = std::fs::remove_file(&dict_path);
        let _ = std::fs::remove_dir(&temp_root);
    }

    #[test]
    fn test_build_char_map_normalizes_ideographic_space_to_ascii_space() {
        let temp_root = std::env::temp_dir().join(format!(
            "aiocr-dataset-test-ideographic-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("main")
        ));
        std::fs::create_dir_all(&temp_root).unwrap();
        let dict_path = temp_root.join("dict.txt");
        std::fs::write(&dict_path, "　\na\nb").unwrap();

        let char_map = OcrDataset::build_char_map(&dict_path).unwrap();

        assert_eq!(char_map.get(&' '), Some(&1));
        assert_eq!(char_map.get(&'a'), Some(&2));
        assert_eq!(char_map.get(&'b'), Some(&3));

        let _ = std::fs::remove_file(&dict_path);
        let _ = std::fs::remove_dir(&temp_root);
    }
}
