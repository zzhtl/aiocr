use crate::config::DetectionBoxMode;
use crate::types::{BoundingBox, ImageMeta};

/// DBNet 后处理参数。
#[derive(Debug, Clone, Copy)]
pub struct DbPostprocessConfig<'a> {
    pub threshold: f32,
    pub box_threshold: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
    /// 最小检测框面积（resized 尺度像素面积），小于此值丢弃。
    pub min_box_area: f32,
    /// 检测框形状模式（轴对齐 / 最小面积旋转矩形）。
    pub box_mode: DetectionBoxMode,
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
    let components = collect_component_stats(&labels, prob_map, width, height);

    // 旋转框模式（仅高精度档）需要每个连通域的像素点集来求最小面积矩形。
    let component_points = match config.box_mode {
        DetectionBoxMode::MinAreaRect => {
            Some(collect_component_points(&labels, width, height, components.len()))
        }
        DetectionBoxMode::AxisAligned => None,
    };

    let mut results = Vec::new();

    for (label, component) in components.iter().enumerate().skip(1) {
        if results.len() >= config.max_candidates {
            break;
        }

        if component.count == 0 {
            continue;
        }

        let mean_score = component.score_sum / component.count as f32;
        if mean_score < config.box_threshold {
            continue;
        }

        // 3. 外接矩形：轴对齐 AABB，或（高精度档）最小面积旋转矩形
        let bbox = match &component_points {
            Some(points) => {
                min_area_rect(&points[label]).unwrap_or_else(|| component.bounding_box())
            }
            None => component.bounding_box(),
        };
        if bbox.area() < config.min_box_area {
            continue;
        }

        // 4. Unclip 扩展（沿框自身轴向，旋转框不退化为 AABB）
        let expanded = unclip_box(&bbox, config.unclip_ratio);

        // 5. 映射回原始图片坐标
        let mapped = map_to_original(&expanded, config.meta);

        results.push((mapped, mean_score));
    }

    sort_reading_order(&mut results);

    results
}

#[derive(Debug, Clone)]
struct ComponentStats {
    x_min: usize,
    y_min: usize,
    x_max: usize,
    y_max: usize,
    score_sum: f32,
    count: usize,
}

impl Default for ComponentStats {
    fn default() -> Self {
        Self {
            x_min: usize::MAX,
            y_min: usize::MAX,
            x_max: 0,
            y_max: 0,
            score_sum: 0.0,
            count: 0,
        }
    }
}

impl ComponentStats {
    fn add_pixel(&mut self, x: usize, y: usize, score: f32) {
        self.x_min = self.x_min.min(x);
        self.y_min = self.y_min.min(y);
        self.x_max = self.x_max.max(x);
        self.y_max = self.y_max.max(y);
        self.score_sum += score;
        self.count += 1;
    }

    fn bounding_box(&self) -> BoundingBox {
        BoundingBox {
            points: [
                [self.x_min as f32, self.y_min as f32],
                [self.x_max as f32, self.y_min as f32],
                [self.x_max as f32, self.y_max as f32],
                [self.x_min as f32, self.y_max as f32],
            ],
        }
    }
}

fn collect_component_stats(
    labels: &[u32],
    prob_map: &[f32],
    width: usize,
    height: usize,
) -> Vec<ComponentStats> {
    let max_label = *labels.iter().max().unwrap_or(&0) as usize;
    let mut components = vec![ComponentStats::default(); max_label + 1];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let label = labels[idx] as usize;
            if label == 0 {
                continue;
            }

            components[label].add_pixel(x, y, prob_map[idx]);
        }
    }

    components
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

/// 扩展检测框（Unclip）
///
/// 采用标准 PaddleOCR/Vatti 距离外扩：`distance = area * ratio / perimeter`。
/// 沿框自身的两条边方向各向外平移 `distance`，因此对旋转框不会退化成轴对齐扩张；
/// 对轴对齐框，其结果与按 to_rect 各边等距外移完全一致（行为不变）。
/// DB 预测的是收缩后的文本核，距离外扩能更完整地恢复行高，贴近识别模型训练分布。
fn unclip_box(bbox: &BoundingBox, ratio: f32) -> BoundingBox {
    let p = &bbox.points;
    let edge_u = [p[1][0] - p[0][0], p[1][1] - p[0][1]];
    let edge_v = [p[3][0] - p[0][0], p[3][1] - p[0][1]];
    let w = (edge_u[0] * edge_u[0] + edge_u[1] * edge_u[1]).sqrt();
    let h = (edge_v[0] * edge_v[0] + edge_v[1] * edge_v[1]).sqrt();

    let perimeter = 2.0 * (w + h);
    if perimeter <= f32::EPSILON {
        return *bbox;
    }
    let distance = w * h * ratio / perimeter;

    // 单位轴向量；某条边退化时回退为坐标轴，避免除零。
    let unit = |edge: [f32; 2], len: f32| {
        if len > f32::EPSILON {
            [edge[0] / len, edge[1] / len]
        } else {
            [0.0, 0.0]
        }
    };
    let ux = unit(edge_u, w);
    let uy = unit(edge_v, h);
    let du = [ux[0] * distance, ux[1] * distance];
    let dv = [uy[0] * distance, uy[1] * distance];

    // p0 向 -u-v，p1 向 +u-v，p2 向 +u+v，p3 向 -u+v 外移。
    BoundingBox {
        points: [
            [p[0][0] - du[0] - dv[0], p[0][1] - du[1] - dv[1]],
            [p[1][0] + du[0] - dv[0], p[1][1] + du[1] - dv[1]],
            [p[2][0] + du[0] + dv[0], p[2][1] + du[1] + dv[1]],
            [p[3][0] - du[0] + dv[0], p[3][1] - du[1] + dv[1]],
        ],
    }
}

/// 收集每个连通域的像素坐标（仅旋转框模式使用）。
fn collect_component_points(
    labels: &[u32],
    width: usize,
    height: usize,
    label_count: usize,
) -> Vec<Vec<[f32; 2]>> {
    let mut points = vec![Vec::new(); label_count];
    for y in 0..height {
        for x in 0..width {
            let label = labels[y * width + x] as usize;
            if label != 0 && label < label_count {
                points[label].push([x as f32, y as f32]);
            }
        }
    }
    points
}

/// 旋转卡壳求最小面积外接矩形，返回顺时针四点（p0→p1 为较长的“宽”边）。
fn min_area_rect(points: &[[f32; 2]]) -> Option<BoundingBox> {
    if points.len() < 3 {
        return None;
    }
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }

    let mut best_area = f32::MAX;
    let mut best: Option<[[f32; 2]; 4]> = None;
    let n = hull.len();
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let len = (edge[0] * edge[0] + edge[1] * edge[1]).sqrt();
        if len <= f32::EPSILON {
            continue;
        }
        let ux = [edge[0] / len, edge[1] / len];
        let uy = [-ux[1], ux[0]];

        let (mut min_x, mut max_x, mut min_y, mut max_y) =
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for h in &hull {
            let px = h[0] * ux[0] + h[1] * ux[1];
            let py = h[0] * uy[0] + h[1] * uy[1];
            min_x = min_x.min(px);
            max_x = max_x.max(px);
            min_y = min_y.min(py);
            max_y = max_y.max(py);
        }

        let area = (max_x - min_x) * (max_y - min_y);
        if area < best_area {
            best_area = area;
            let corner = |px: f32, py: f32| [px * ux[0] + py * uy[0], px * ux[1] + py * uy[1]];
            let (bw, bh) = (max_x - min_x, max_y - min_y);
            // 让 p0→p1 为较长的“宽”边，使裁剪结果尽量为水平文本。
            best = Some(if bw >= bh {
                [
                    corner(min_x, min_y),
                    corner(max_x, min_y),
                    corner(max_x, max_y),
                    corner(min_x, max_y),
                ]
            } else {
                [
                    corner(min_x, max_y),
                    corner(min_x, min_y),
                    corner(max_x, min_y),
                    corner(max_x, max_y),
                ]
            });
        }
    }

    best.map(|points| BoundingBox { points })
}

/// Andrew monotone chain 凸包，返回逆时针顶点。
fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a[0]
            .partial_cmp(&b[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| a[0] == b[0] && a[1] == b[1]);
    if pts.len() < 3 {
        return pts;
    }

    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };

    let mut hull: Vec<[f32; 2]> = Vec::with_capacity(pts.len() + 1);
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
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

    #[test]
    fn test_unclip_box_axis_aligned_matches_distance_offset() {
        // 旋转感知的 unclip 对轴对齐框应等价于各边等距外移（行为不变）。
        let bbox = BoundingBox {
            points: [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]],
        };
        // w=40, h=20, distance = 40*20*1.5/120 = 10
        let out = unclip_box(&bbox, 1.5);
        assert!((out.points[0][0] - 0.0).abs() < 1e-3);
        assert!((out.points[0][1] - 0.0).abs() < 1e-3);
        assert!((out.points[2][0] - 60.0).abs() < 1e-3);
        assert!((out.points[2][1] - 40.0).abs() < 1e-3);
    }

    #[test]
    fn test_min_area_rect_axis_aligned_rectangle() {
        let pts = vec![
            [10.0, 10.0],
            [50.0, 10.0],
            [50.0, 30.0],
            [10.0, 30.0],
            [30.0, 20.0],
        ];
        let rect = min_area_rect(&pts).expect("应得到矩形");
        assert!((rect.width() - 40.0).abs() < 1.0, "w={}", rect.width());
        assert!((rect.height() - 20.0).abs() < 1.0, "h={}", rect.height());
        assert!((rect.area() - 800.0).abs() < 5.0);
    }

    #[test]
    fn test_min_area_rect_recovers_45deg_rectangle() {
        // 40x20 矩形绕中心旋转 45°（中心平移到 (30,30)）。
        let pts = vec![
            [37.07, 51.21],
            [51.21, 37.07],
            [22.93, 8.79],
            [8.79, 22.93],
        ];
        let rect = min_area_rect(&pts).expect("应得到矩形");
        let (mut w, mut h) = (rect.width(), rect.height());
        if w < h {
            std::mem::swap(&mut w, &mut h);
        }
        assert!((w - 40.0).abs() < 1.5, "w={w}");
        assert!((h - 20.0).abs() < 1.5, "h={h}");
        assert!((rect.area() - 800.0).abs() < 12.0, "area={}", rect.area());
    }

    #[test]
    fn test_is_axis_aligned_detects_rotation() {
        assert!(BoundingBox::from_rect(10.0, 10.0, 40.0, 20.0).is_axis_aligned());
        let rotated = BoundingBox {
            points: [[37.07, 51.21], [51.21, 37.07], [22.93, 8.79], [8.79, 22.93]],
        };
        assert!(!rotated.is_axis_aligned());
    }
}
