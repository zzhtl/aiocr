use egui::Ui;

use crate::state::TrainingState;
use crate::widgets::progress;

/// 训练面板动作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingAction {
    None,
    PickDataset,
    PickArtifactDir,
    StartTraining,
    CancelTraining,
    RefreshModels,
    UseSelectedModel,
}

/// 训练面板。
pub fn show(ui: &mut Ui, training: &mut TrainingState, is_busy: bool) -> TrainingAction {
    let mut action = TrainingAction::None;
    let base_model_name = training
        .active_model()
        .map(|model| model.display_name())
        .unwrap_or_else(|| "默认 PP-OCRv5 Server（models/）".to_string());

    ui.heading("模型训练与微调");
    ui.separator();

    ui.label(format!("基础模型: {base_model_name}"));
    ui.label(
        egui::RichText::new(
            "在已有模型基础上继续训练，使用您标注的数据集进行微调，提升特定场景识别精度。",
        )
        .small()
        .weak(),
    );

    ui.add_space(6.0);

    // 数据集目录
    ui.label("训练数据集:");
    ui.horizontal(|ui| {
        let mut path = training.config.dataset_path.display().to_string();
        ui.add_enabled(
            false,
            egui::TextEdit::singleline(&mut path).desired_width(400.0),
        );
        if ui
            .add_enabled(!is_busy, egui::Button::new("选择"))
            .clicked()
        {
            action = TrainingAction::PickDataset;
        }
    });
    ui.label(
        egui::RichText::new(
            "数据集格式: dataset/images/*.jpg + dataset/labels.txt (文件名\\t文字)",
        )
        .small()
        .weak(),
    );

    ui.add_space(4.0);

    // 产物目录
    ui.label("模型输出目录:");
    ui.horizontal(|ui| {
        let mut path = training.config.artifact_dir.display().to_string();
        ui.add_enabled(
            false,
            egui::TextEdit::singleline(&mut path).desired_width(400.0),
        );
        if ui
            .add_enabled(!is_busy, egui::Button::new("选择"))
            .clicked()
        {
            action = TrainingAction::PickArtifactDir;
        }
    });

    ui.separator();

    // 超参数
    ui.label("训练参数:");
    ui.horizontal(|ui| {
        ui.label("轮数(Epoch)");
        ui.add(egui::DragValue::new(&mut training.config.num_epochs).range(1..=500));
        ui.separator();
        ui.label("批大小(Batch)");
        ui.add(egui::DragValue::new(&mut training.config.batch_size).range(1..=512));
        ui.separator();
        ui.label("学习率(LR)");
        ui.add(
            egui::DragValue::new(&mut training.config.learning_rate)
                .range(1e-6..=0.1)
                .speed(1e-5)
                .min_decimals(6),
        );
    });

    ui.horizontal(|ui| {
        ui.label("输入高度");
        ui.add(egui::DragValue::new(&mut training.config.img_height).range(16..=128));
        ui.separator();
        ui.label("输入宽度");
        ui.add(egui::DragValue::new(&mut training.config.img_width).range(32..=1024));
    });

    ui.label(
        egui::RichText::new("训练中会自动应用数据增强（亮度/对比度/噪声/模糊）提升泛化能力。")
            .small()
            .weak(),
    );

    ui.separator();

    // 训练控制
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!is_busy, egui::Button::new("🚀 开始训练"))
            .clicked()
        {
            action = TrainingAction::StartTraining;
        }
        if ui
            .add_enabled(is_busy, egui::Button::new("⏹ 停止训练"))
            .clicked()
        {
            action = TrainingAction::CancelTraining;
        }
        if ui
            .add_enabled(!is_busy, egui::Button::new("🔄 刷新列表"))
            .clicked()
        {
            action = TrainingAction::RefreshModels;
        }
    });

    // 训练进度
    if let Some(prog) = &training.progress {
        ui.separator();
        progress::show_progress(ui, prog);
    }

    ui.separator();
    ui.heading("本地 AI 模型管理");

    // 当前激活模型
    ui.horizontal(|ui| {
        ui.label("当前激活:");
        if let Some(model) = training.active_model() {
            ui.label(egui::RichText::new(model.display_name()).strong());
        } else {
            ui.label(egui::RichText::new("默认 PP-OCRv5 Server").weak());
        }
    });

    // 使用选中模型按钮
    ui.horizontal(|ui| {
        let has_selection = training.selected_model().is_some();
        if ui
            .add_enabled(
                has_selection && !is_busy,
                egui::Button::new("切换到选中模型"),
            )
            .clicked()
        {
            action = TrainingAction::UseSelectedModel;
        }

        if let Some(model) = training.selected_model() {
            ui.label(format!("→ {}", model.display_name()));
        }
    });

    // 模型列表
    egui::ScrollArea::vertical()
        .max_height(200.0)
        .show(ui, |ui| {
            if training.available_models.is_empty() {
                ui.label(egui::RichText::new("暂无本地训练模型").weak());
                return;
            }

            for model in &training.available_models {
                let selected = training
                    .selected_model_dir
                    .as_ref()
                    .is_some_and(|path| path == &model.model_dir);

                let label = format!(
                    "{} | 样本 {}/{} | loss {:.4}",
                    model.display_name(),
                    model.train_samples,
                    model.validation_samples,
                    model.avg_loss,
                );
                let response = ui.selectable_label(selected, label);
                if response.clicked() {
                    training.selected_model_dir = Some(model.model_dir.clone());
                }
                ui.label(
                    egui::RichText::new(format!("基于: {}", model.base_model_name))
                        .small()
                        .weak(),
                );
                ui.separator();
            }
        });

    action
}
