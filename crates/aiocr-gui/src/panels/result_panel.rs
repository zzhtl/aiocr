use aiocr_core::build_spatial_text;
use aiocr_core::types::TextDirection;
use egui::{RichText, ScrollArea, TextEdit, Ui};

use crate::state::AppState;

const MIN_FULL_TEXT_HEIGHT: f32 = 180.0;
const MAX_FULL_TEXT_HEIGHT: f32 = 320.0;

/// OCR 结果展示面板
pub fn show(ui: &mut Ui, state: &AppState) {
    ui.heading("识别结果");
    ui.separator();

    if let Some(result) = &state.ocr_result {
        let spatial_text = state
            .image_data
            .as_ref()
            .map(|image| {
                build_spatial_text(
                    &result.regions,
                    image.width() as f32,
                    image.height() as f32,
                )
            })
            .unwrap_or_else(|| result.full_text.clone());

        ui.horizontal_wrapped(|ui| {
            ui.label(format!("检测到 {} 个文本区域", result.regions.len()));
            ui.separator();
            ui.label(format!("耗时 {}ms", result.elapsed_ms));
            ui.separator();

            if ui.button("复制版式").clicked() {
                ui.ctx().copy_text(spatial_text.clone());
            }
            if ui.button("复制全部").clicked() {
                ui.ctx().copy_text(result.full_text.clone());
            }
        });
        ui.add_space(8.0);

        ui.label(RichText::new("版式预览").strong());
        let mut layout_text = spatial_text;
        let available_height = ui.available_height();
        let layout_text_height = (available_height * 0.48)
            .min(MAX_FULL_TEXT_HEIGHT)
            .max(MIN_FULL_TEXT_HEIGHT.min(available_height * 0.6));
        ScrollArea::both()
            .id_salt("ocr_spatial_text_scroll")
            .max_height(layout_text_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add(
                    TextEdit::multiline(&mut layout_text)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace)
                        .code_editor(),
                );
            });

        ui.add_space(6.0);
        egui::CollapsingHeader::new("完整文本")
            .default_open(false)
            .show(ui, |ui| {
                let mut text = result.full_text.clone();
                ScrollArea::both()
                    .id_salt("ocr_full_text_scroll")
                    .max_height(140.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add(
                            TextEdit::multiline(&mut text)
                                .desired_width(f32::INFINITY)
                                .font(egui::TextStyle::Monospace)
                                .code_editor(),
                        );
                    });
            });

        ui.add_space(8.0);
        ui.label(RichText::new("分段结果").strong());

        let detail_list_height = ui.available_height().max(0.0);
        ScrollArea::vertical()
            .id_salt("ocr_result_regions")
            .auto_shrink([false, false])
            .max_height(detail_list_height)
            .show(ui, |ui| {
                for (i, region) in result.regions.iter().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(format!("#{}", i + 1));
                            ui.separator();
                            ui.label(format!("置信度 {:.0}%", region.confidence * 100.0));
                            ui.separator();
                            ui.label(direction_label(region.direction));
                        });
                        ui.add_space(4.0);
                        ui.add(egui::Label::new(RichText::new(&region.text).monospace()).wrap());
                    });
                    ui.add_space(6.0);
                }
            });
    } else {
        ui.label("暂无识别结果");
        ui.label("请先打开图片、拖拽图片，或按 Ctrl+V / Cmd+V 粘贴截图后开始识别");
    }
}

fn direction_label(direction: TextDirection) -> &'static str {
    match direction {
        TextDirection::Horizontal => "横排",
        TextDirection::Vertical => "竖排",
    }
}
