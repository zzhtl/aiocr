use crate::types::{BoundingBox, ImageMeta};

/// DBNet 后处理参数。
#[derive(Debug, Clone, Copy)]
pub struct DbPostprocessConfig<'a> {
    pub threshold: f32,
    pub box_threshold: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
    pub meta: &'a ImageMeta,
}

/// DBNet 后处理：从概率图提取文本检测框
pub fn db_postprocess(
    prob_map: &[f32],
    height: usize,
    width: usize,
    config: DbPostprocessConfig<'_>,
) -> Vec<(BoundingBox, f32)> {
    // 1. 二值化
    let binary: Vec<u8> = prob_map
        .iter()
        .map(|&v| if v > config.threshold { 1 } else { 0 })
        .collect();

    // 2. 连通域分析
    let labels = connected_components(&binary, width, height);
    let num_labels = *labels.iter().max().unwrap_or(&0);

    let mut results = Vec::new();

    for label in 1..=num_labels {
        if results.len() >= config.max_candidates {
            break;
        }

        // 收集当前连通域的像素坐标
        let mut points: Vec<[usize; 2]> = Vec::new();
        let mut score_sum = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if labels[idx] == label {
                    points.push([x, y]);
                    score_sum += prob_map[idx];
                }
            }
        }

        if points.is_empty() {
            continue;
        }

        let mean_score = score_sum / points.len() as f32;
        if mean_score < config.box_threshold {
            continue;
        }

        // 3. 最小外接矩形
        let bbox = min_bounding_rect(&points);
        if bbox.area() < 10.0 {
            continue;
        }

        // 4. Unclip 扩展
        let expanded = unclip_box(&bbox, config.unclip_ratio);

        // 5. 映射回原始图片坐标
        let mapped = map_to_original(&expanded, config.meta);

        results.push((mapped, mean_score));
    }

    sort_reading_order(&mut results);

    results
}

/// 简单的连通域标记（4-连通）
fn connected_components(binary: &[u8], width: usize, height: usize) -> Vec<u32> {
    let mut labels = vec![0u32; width * height];
    let mut next_label = 1u32;
    let mut equivalences: Vec<u32> = vec![0]; // equivalences[label] = root

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if binary[idx] == 0 {
                continue;
            }

            let left = if x > 0 { labels[idx - 1] } else { 0 };
            let top = if y > 0 { labels[idx - width] } else { 0 };

            match (left > 0, top > 0) {
                (false, false) => {
                    labels[idx] = next_label;
                    equivalences.push(next_label);
                    next_label += 1;
                }
                (true, false) => labels[idx] = left,
                (false, true) => labels[idx] = top,
                (true, true) => {
                    let min_label = left.min(top);
                    let max_label = left.max(top);
                    labels[idx] = min_label;
                    // 合并等价类
                    let root_max = find_root(&equivalences, max_label);
                    let root_min = find_root(&equivalences, min_label);
                    if root_max != root_min {
                        equivalences[root_max as usize] = root_min;
                    }
                }
            }
        }
    }

    // 第二遍：统一标签
    for label in labels.iter_mut() {
        if *label > 0 {
            *label = find_root(&equivalences, *label);
        }
    }

    labels
}

/// 查找等价类根节点
fn find_root(equivalences: &[u32], mut label: u32) -> u32 {
    while equivalences[label as usize] != label {
        label = equivalences[label as usize];
    }
    label
}

/// 从点集计算最小外接矩形（轴对齐）
fn min_bounding_rect(points: &[[usize; 2]]) -> BoundingBox {
    let mut x_min = usize::MAX;
    let mut y_min = usize::MAX;
    let mut x_max = 0usize;
    let mut y_max = 0usize;

    for p in points {
        x_min = x_min.min(p[0]);
        y_min = y_min.min(p[1]);
        x_max = x_max.max(p[0]);
        y_max = y_max.max(p[1]);
    }

    BoundingBox {
        points: [
            [x_min as f32, y_min as f32],
            [x_max as f32, y_min as f32],
            [x_max as f32, y_max as f32],
            [x_min as f32, y_max as f32],
        ],
    }
}

/// 扩展检测框（Unclip）
fn unclip_box(bbox: &BoundingBox, ratio: f32) -> BoundingBox {
    let rect = bbox.to_rect();
    let w = rect[2] - rect[0];
    let h = rect[3] - rect[1];
    let expand_w = w * (ratio - 1.0) / 2.0;
    let expand_h = h * (ratio - 1.0) / 2.0;

    BoundingBox {
        points: [
            [rect[0] - expand_w, rect[1] - expand_h],
            [rect[2] + expand_w, rect[1] - expand_h],
            [rect[2] + expand_w, rect[3] + expand_h],
            [rect[0] - expand_w, rect[3] + expand_h],
        ],
    }
}

/// 将检测框坐标映射回原始图片尺寸
fn map_to_original(bbox: &BoundingBox, meta: &ImageMeta) -> BoundingBox {
    let mut mapped = *bbox;
    for point in &mut mapped.points {
        let x = (point[0] - meta.pad_x as f32).clamp(0.0, meta.content_width as f32);
        let y = (point[1] - meta.pad_y as f32).clamp(0.0, meta.content_height as f32);
        point[0] = (x * meta.scale_x).clamp(0.0, meta.orig_width as f32);
        point[1] = (y * meta.scale_y).clamp(0.0, meta.orig_height as f32);
    }
    mapped
}

fn sort_reading_order(results: &mut [(BoundingBox, f32)]) {
    results.sort_by(|a, b| {
        let ay = a.0.top();
        let by = b.0.top();
        let avg_height = (a.0.height() + b.0.height()) * 0.5;
        let same_line = (ay - by).abs() <= avg_height * 0.5;

        if same_line {
            a.0.left()
                .partial_cmp(&b.0.left())
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            ay.partial_cmp(&by).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unclip_box() {
        let bbox = BoundingBox {
            points: [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]],
        };
        let expanded = unclip_box(&bbox, 1.5);
        assert!(expanded.points[0][0] < 10.0);
        assert!(expanded.points[2][0] > 50.0);
    }

    #[test]
    fn test_connected_components_single_region() {
        #[rustfmt::skip]
        let binary = vec![
            0, 0, 0, 0, 0,
            0, 1, 1, 0, 0,
            0, 1, 1, 0, 0,
            0, 0, 0, 0, 0,
        ];
        let labels = connected_components(&binary, 5, 4);
        let unique: std::collections::HashSet<u32> =
            labels.iter().filter(|&&l| l > 0).copied().collect();
        assert_eq!(unique.len(), 1);
    }
}
