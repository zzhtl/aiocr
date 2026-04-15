use std::path::Path;

use crate::error::OcrError;

/// CTC 贪心解码器
pub struct CtcDecoder {
    /// 字符字典（index -> 字符）
    char_dict: Vec<String>,
}

impl CtcDecoder {
    /// 从字典文件加载
    pub fn from_dict_file(path: &Path) -> Result<Self, OcrError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| OcrError::Dictionary(format!("{path:?}: {e}")))?;

        // PaddleOCR 字典格式：每行一个字符，index 0 保留给 blank token。
        // 对仅包含全角空格（U+3000）的官方字典，统一映射为 ASCII 空格，
        // 便于和常见标签文本保持一致，同时不改变类别总数。
        let entries: Vec<&str> = content
            .lines()
            .map(|line| line.strip_suffix('\r').unwrap_or(line))
            .filter(|line| !line.is_empty())
            .collect();
        let has_ascii_space = entries.iter().any(|entry| *entry == " ");

        let mut char_dict = vec!["<blank>".to_string()];
        for entry in entries {
            let normalized = if !has_ascii_space && entry == "\u{3000}" {
                " "
            } else {
                entry
            };
            char_dict.push(normalized.to_string());
        }

        tracing::info!("加载字典: {} 个字符", char_dict.len());

        Ok(Self { char_dict })
    }

    /// 尝试从文件加载字典；文件缺失时回退到内置 ASCII 字符集。
    pub fn from_dict_or_builtin(path: &Path) -> Result<Self, OcrError> {
        match Self::from_dict_file(path) {
            Ok(decoder) => Ok(decoder),
            Err(err) => {
                tracing::warn!("字典加载失败，使用内置字典: {err}");
                Ok(Self::builtin_ascii())
            }
        }
    }

    /// 内置最小字典，确保在未提供 PaddleOCR 字典时主流程仍可初始化。
    pub fn builtin_ascii() -> Self {
        let mut char_dict = vec!["<blank>".to_string()];
        let builtin = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;!?+-*/=_()[]{}<>@#%&'\" ";
        for ch in builtin.chars() {
            char_dict.push(ch.to_string());
        }
        Self { char_dict }
    }

    /// CTC 贪心解码
    ///
    /// 输入: 模型输出的 logits [T, num_classes]
    /// 返回: (解码文本, 平均置信度)
    pub fn decode(&self, logits: &[f32], num_classes: usize) -> (String, f32) {
        self.decode_with(logits, num_classes, softmax_argmax)
    }

    /// 对已经归一化过的类别概率执行 CTC 贪心解码。
    pub fn decode_probabilities(&self, probabilities: &[f32], num_classes: usize) -> (String, f32) {
        self.decode_with(probabilities, num_classes, probability_argmax)
    }

    fn decode_with<F>(&self, values: &[f32], num_classes: usize, pick: F) -> (String, f32)
    where
        F: Fn(&[f32]) -> (usize, f32),
    {
        let time_steps = values.len() / num_classes;
        let mut text = String::new();
        let mut prev_idx = 0usize;
        let mut total_conf = 0.0f32;
        let mut char_count = 0usize;

        for t in 0..time_steps {
            let offset = t * num_classes;
            let slice = &values[offset..offset + num_classes];
            let (max_idx, max_val) = pick(slice);

            // CTC 规则：跳过 blank (0) 和重复字符
            if max_idx != 0
                && max_idx != prev_idx
                && let Some(ch) = self.char_dict.get(max_idx)
            {
                text.push_str(ch);
                total_conf += max_val;
                char_count += 1;
            }
            prev_idx = max_idx;
        }

        let avg_conf = if char_count > 0 {
            total_conf / char_count as f32
        } else {
            0.0
        };

        (text, avg_conf)
    }

    /// 字典大小（含 blank）
    pub fn num_classes(&self) -> usize {
        self.char_dict.len()
    }
}

/// Softmax 后取 argmax，返回 (index, probability)
fn softmax_argmax(logits: &[f32]) -> (usize, f32) {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut max_idx = 0;
    let mut max_prob = 0.0f32;

    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    for (i, &e) in exps.iter().enumerate() {
        let prob = e / sum;
        if prob > max_prob {
            max_prob = prob;
            max_idx = i;
        }
    }

    (max_idx, max_prob)
}

fn probability_argmax(probabilities: &[f32]) -> (usize, f32) {
    let mut max_idx = 0usize;
    let mut max_prob = f32::NEG_INFINITY;

    for (idx, &prob) in probabilities.iter().enumerate() {
        if prob > max_prob {
            max_prob = prob;
            max_idx = idx;
        }
    }

    (max_idx, max_prob.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_argmax() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let (idx, prob) = softmax_argmax(&logits);
        assert_eq!(idx, 1);
        assert!(prob > 0.5);
    }

    #[test]
    fn test_ctc_decode_removes_blanks_and_duplicates() {
        let decoder = CtcDecoder {
            char_dict: vec![
                "<blank>".to_string(),
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
            ],
        };

        // 模拟 logits: blank, a, a, b, blank, c
        // 每个 timestep 4 个 class
        let logits = vec![
            10.0, 0.0, 0.0, 0.0, // blank
            0.0, 10.0, 0.0, 0.0, // a
            0.0, 10.0, 0.0, 0.0, // a (重复，应跳过)
            0.0, 0.0, 10.0, 0.0, // b
            10.0, 0.0, 0.0, 0.0, // blank
            0.0, 0.0, 0.0, 10.0, // c
        ];

        let (text, _conf) = decoder.decode(&logits, 4);
        assert_eq!(text, "abc");
    }

    #[test]
    fn test_dict_loader_preserves_space_token_without_duplicate_append() {
        let temp_root = std::env::temp_dir().join(format!(
            "aiocr-decoder-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("main")
        ));
        std::fs::create_dir_all(&temp_root).unwrap();
        let dict_path = temp_root.join("dict.txt");
        std::fs::write(&dict_path, "a\n \nb").unwrap();

        let decoder = CtcDecoder::from_dict_file(&dict_path).unwrap();

        assert_eq!(decoder.num_classes(), 4);

        let _ = std::fs::remove_file(&dict_path);
        let _ = std::fs::remove_dir(&temp_root);
    }

    #[test]
    fn test_dict_loader_normalizes_ideographic_space_to_ascii_space() {
        let temp_root = std::env::temp_dir().join(format!(
            "aiocr-decoder-test-ideographic-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("main")
        ));
        std::fs::create_dir_all(&temp_root).unwrap();
        let dict_path = temp_root.join("dict.txt");
        std::fs::write(&dict_path, "　\na\nb").unwrap();

        let decoder = CtcDecoder::from_dict_file(&dict_path).unwrap();
        let (text, _conf) = decoder.decode_probabilities(&[0.0, 1.0, 0.0, 0.0], 4);

        assert_eq!(decoder.num_classes(), 4);
        assert_eq!(text, " ");

        let _ = std::fs::remove_file(&dict_path);
        let _ = std::fs::remove_dir(&temp_root);
    }
}
