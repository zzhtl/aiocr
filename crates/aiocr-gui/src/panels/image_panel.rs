use egui::Ui;

use crate::state::AppState;
use crate::widgets::bbox_overlay;

/// 图片预览面板
pub fn show(ui: &mut Ui, state: &AppState, texture: Option<&egui::TextureHandle>) {
    egui::ScrollArea::both().show(ui, |ui| {
        if let Some(tex) = texture {
            let size = tex.size_vec2();
            // 适配面板宽度
            let available_width = ui.available_width();
            let scale = (available_width / size.x).min(1.0);
            let display_size = egui::vec2(size.x * scale, size.y * scale);

            let response = ui.image(egui::load::SizedTexture::new(tex.id(), display_size));

            // 绘制检测框
            if state.show_bboxes
                && let Some(result) = &state.ocr_result
            {
                let painter = ui.painter_at(response.rect);
                draw_bboxes(
                    &painter,
                    &response.rect,
                    &result.regions,
                    &display_size,
                    state,
                );
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("拖拽图片到此处，点击“打开图片”，或按 Ctrl+V / Cmd+V 粘贴截图");
            });
        }
    });
}

/// 在图片上绘制检测框
fn draw_bboxes(
    painter: &egui::Painter,
    image_rect: &egui::Rect,
    regions: &[aiocr_core::types::TextRegion],
    display_size: &egui::Vec2,
    state: &AppState,
) {
    let Some(img) = &state.image_data else {
        return;
    };

    let img_w = img.width() as f32;
    let img_h = img.height() as f32;
    let scale_x = display_size.x / img_w;
    let scale_y = display_size.y / img_h;

    for region in regions {
        bbox_overlay::draw_bbox(
            painter,
            &region.bbox,
            image_rect.min,
            scale_x,
            scale_y,
            egui::Color32::from_rgb(0, 200, 0),
        );
    }
}
