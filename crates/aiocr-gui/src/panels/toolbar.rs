use egui::Ui;

use crate::state::AppState;

/// 顶部工具栏
pub fn show(ui: &mut Ui, state: &AppState) -> ToolbarAction {
    let mut action = ToolbarAction::None;

    ui.horizontal(|ui| {
        if ui
            .add_enabled(!state.task_status.is_busy(), egui::Button::new("打开图片"))
            .clicked()
        {
            action = ToolbarAction::OpenImage;
        }

        if ui
            .add_enabled(!state.task_status.is_busy(), egui::Button::new("粘贴截图"))
            .on_hover_text("从系统剪贴板读取截图并直接开始识别（Ctrl+V / Cmd+V）")
            .clicked()
        {
            action = ToolbarAction::PasteImage;
        }

        if ui
            .add_enabled(
                state.image_data.is_some() && !state.task_status.is_busy(),
                egui::Button::new("开始识别"),
            )
            .clicked()
        {
            action = ToolbarAction::StartOcr;
        }

        ui.separator();

        if state.task_status.is_busy() {
            ui.spinner();
            ui.label("处理中...");
        } else if let Some(model) = state.training.active_model() {
            ui.label(format!("当前模型: {}", model.display_name()));
        } else {
            ui.label("当前模型: 默认 PP-OCRv5 Server");
        }
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
}
