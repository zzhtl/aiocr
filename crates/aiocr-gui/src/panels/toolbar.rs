use egui::{Color32, Layout, RichText, Ui};

use crate::state::AppState;
use crate::theme;

/// 顶部工具栏
pub fn show(ui: &mut Ui, state: &AppState) -> ToolbarAction {
    let mut action = ToolbarAction::None;
    let busy = state.task_status.is_busy();

    ui.horizontal(|ui| {
        ui.add_space(2.0);
        ui.label(RichText::new("AIOCR").strong().size(16.0));
        ui.separator();

        if ui
            .add_enabled(!busy, egui::Button::new("打开图片"))
            .clicked()
        {
            action = ToolbarAction::OpenImage;
        }

        if ui
            .add_enabled(!busy, egui::Button::new("粘贴截图"))
            .on_hover_text("从系统剪贴板读取截图并直接开始识别（Ctrl+V / Cmd+V）")
            .clicked()
        {
            action = ToolbarAction::PasteImage;
        }

        let accent = theme::accent(ui.visuals());
        let start = egui::Button::new(RichText::new("▶ 开始识别").color(Color32::WHITE)).fill(accent);
        if ui
            .add_enabled(state.image_data.is_some() && !busy, start)
            .clicked()
        {
            action = ToolbarAction::StartOcr;
        }

        // 右侧：主题切换 + 当前模型/状态
        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
            let mode = state.theme_mode;
            if ui
                .button(mode.icon())
                .on_hover_text(format!("主题：{}（点击切换）", mode.label()))
                .clicked()
            {
                action = ToolbarAction::ToggleTheme;
            }

            ui.separator();

            if busy {
                ui.label("处理中...");
                ui.spinner();
            } else if let Some(model) = state.training.active_model() {
                ui.label(format!("模型: {}", model.display_name()));
            } else {
                ui.label("模型: 默认 PP-OCRv5 Server");
            }
        });
    });

    action
}

/// 工具栏操作
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolbarAction {
    None,
    OpenImage,
    PasteImage,
    StartOcr,
    ToggleTheme,
}
