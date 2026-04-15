use egui::Painter;

use aiocr_core::types::BoundingBox;

/// 在 Painter 上绘制检测框
pub fn draw_bbox(
    painter: &Painter,
    bbox: &BoundingBox,
    offset: egui::Pos2,
    scale_x: f32,
    scale_y: f32,
    color: egui::Color32,
) {
    let points: Vec<egui::Pos2> = bbox
        .points
        .iter()
        .map(|p| egui::pos2(offset.x + p[0] * scale_x, offset.y + p[1] * scale_y))
        .collect();

    let stroke = egui::Stroke::new(2.0, color);
    for i in 0..4 {
        painter.line_segment([points[i], points[(i + 1) % 4]], stroke);
    }
}
