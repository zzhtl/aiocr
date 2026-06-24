use aiocr_core::QualityPreset;
use egui::Ui;

use crate::state::{AppState, EngineBackend};
use crate::theme;

/// 控制面板动作
pub enum ControlAction {
    None,
    /// 切换引擎后端
    SwitchBackend(EngineBackend),
    /// 选择 ONNX 模型目录
    PickOnnxDir,
    /// 切换识别质量预设
    SetPreset(QualityPreset),
}

/// 控制面板：显示当前模型状态，支持切换引擎后端
pub fn show(ui: &mut Ui, state: &mut AppState) -> ControlAction {
    let mut action = ControlAction::None;

    ui.heading("识别设置");
    ui.add_space(2.0);

    // 识别质量预设
    ui.label("识别质量:");
    ui.horizontal(|ui| {
        for preset in [
            QualityPreset::Fast,
            QualityPreset::Balanced,
            QualityPreset::High,
        ] {
            let selected = state.quality_preset == preset;
            if ui
                .selectable_label(selected, preset.display_name())
                .on_hover_text(preset_hint(preset))
                .clicked()
                && !selected
            {
                action = ControlAction::SetPreset(preset);
            }
        }
    });

    ui.add_space(4.0);
    ui.checkbox(&mut state.show_bboxes, "显示检测框");
    ui.separator();

    // 模型后端选择（纵向排列，避免长文本换行）
    ui.label("识别引擎:");
    for backend in [
        EngineBackend::BurnDefault,
        EngineBackend::Onnx,
        EngineBackend::LocalAi,
    ] {
        let selected = state.active_backend == backend;
        if ui
            .selectable_label(selected, backend.display_name())
            .clicked()
            && !selected
        {
            action = ControlAction::SwitchBackend(backend);
        }
    }

    ui.separator();

    // 根据后端显示对应配置区域
    match &state.active_backend {
        EngineBackend::BurnDefault => {
            show_burn_info(ui);
        }
        EngineBackend::Onnx => {
            if let ControlAction::None = &action {
                action = show_onnx_config(ui, state);
            } else {
                show_onnx_config(ui, state);
            }
        }
        EngineBackend::LocalAi => {
            show_local_ai_info(ui, state);
        }
    }

    ui.separator();
    show_image_info(ui, state);

    action
}

fn show_burn_info(ui: &mut Ui) {
    egui::Grid::new("burn_info_grid")
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            ui.label("模型");
            ui.label("PP-OCRv5 Server（默认混合链路）");
            ui.end_row();

            ui.label("检测");
            ui.label("优先 ONNX，缺失时回退 Burn");
            ui.end_row();

            ui.label("识别");
            ui.label("优先 Burn，失败时回退 ONNX");
            ui.end_row();

            ui.label("方向分类");
            ui.label("现有 2-class cls");
            ui.end_row();
        });

    ui.add_space(4.0);
    ui.label(
        egui::RichText::new("提示：默认链路会优先用 server ONNX 做检测、用 Burn 做识别；这样能保住训练链路，同时避免 server det 在纯 Burn CPU 上过慢。\n如需进一步提升效果，可在「模型训练」页继续微调，或选择「外部 ONNX 模型」加载其他模型。")
            .small()
            .weak(),
    );
}

fn show_onnx_config(ui: &mut Ui, state: &AppState) -> ControlAction {
    let mut action = ControlAction::None;

    egui::Grid::new("onnx_config_grid")
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            ui.label("det.onnx");
            if let Some(p) = &state.onnx.det_path {
                ui.label(
                    p.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                )
                .on_hover_text(p.display().to_string());
            } else {
                ui.label(egui::RichText::new("未设置").weak());
            }
            ui.end_row();

            ui.label("rec.onnx");
            if let Some(p) = &state.onnx.rec_path {
                ui.label(
                    p.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                )
                .on_hover_text(p.display().to_string());
            } else {
                ui.label(egui::RichText::new("未设置").weak());
            }
            ui.end_row();

            ui.label("字典");
            if let Some(p) = &state.onnx.dict_path {
                ui.label(
                    p.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                )
                .on_hover_text(p.display().to_string());
            } else {
                ui.label(egui::RichText::new("使用内置字典").weak());
            }
            ui.end_row();
        });

    ui.add_space(4.0);

    if ui.button("选择 ONNX 模型目录").clicked() {
        action = ControlAction::PickOnnxDir;
    }

    if !state.onnx.is_usable() {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(
                "需要包含 det.onnx 和 rec.onnx 的目录。\n建议优先使用 PP-OCRv5 Server ONNX，或将官方 Paddle Inference 模型导出为 ONNX。"
            )
            .small()
            .color(theme::warning(ui.visuals())),
        );
    } else {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("ONNX 模型已就绪")
                .small()
                .color(theme::success(ui.visuals())),
        );
    }

    action
}

fn preset_hint(preset: QualityPreset) -> &'static str {
    match preset {
        QualityPreset::Fast => "较低检测分辨率，优先速度",
        QualityPreset::Balanced => "兼顾速度与精度",
        QualityPreset::High => "较高检测分辨率，优先识别效果（较慢）",
    }
}

fn show_local_ai_info(ui: &mut Ui, state: &AppState) {
    egui::Grid::new("ai_info_grid")
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            ui.label("当前模型");
            if let Some(model) = state.training.active_model() {
                ui.label(format!("{}", model.display_name()))
                    .on_hover_text(format!("基于: {}", model.base_model_name));
            } else {
                ui.label(egui::RichText::new("未选择模型").weak());
            }
            ui.end_row();
        });

    if state.training.active_model().is_none() {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("请在「模型训练」页训练并切换到本地 AI 模型。")
                .small()
                .weak(),
        );
    }
}

fn show_image_info(ui: &mut Ui, state: &AppState) {
    egui::Grid::new("image_info_grid")
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            ui.label("图片状态");
            if let Some(img) = &state.image_data {
                ui.label(format!("{}×{} px", img.width(), img.height()));
            } else {
                ui.label(egui::RichText::new("未加载").weak());
            }
            ui.end_row();

            ui.label("图片来源");
            if let Some(path) = &state.image_path {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                ui.label(name).on_hover_text(path.display().to_string());
            } else if state.image_data.is_some() {
                ui.label("剪贴板");
            } else {
                ui.label(egui::RichText::new("-").weak());
            }
            ui.end_row();
        });
}
