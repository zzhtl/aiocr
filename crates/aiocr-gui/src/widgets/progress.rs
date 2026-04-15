use aiocr_train::TrainingProgress;
use egui::Ui;

/// 训练进度条组件
pub fn show_progress(ui: &mut Ui, progress: &TrainingProgress) {
    let total_epochs = progress.total_epochs.max(1);
    let overall = ((progress.epoch.saturating_sub(1)) as f32
        + progress.batch as f32 / progress.total_batches.max(1) as f32)
        / total_epochs as f32;

    ui.add(
        egui::ProgressBar::new(overall.clamp(0.0, 1.0)).text(format!(
            "Epoch {}/{}  Batch {}/{}",
            progress.epoch, progress.total_epochs, progress.batch, progress.total_batches
        )),
    );
    ui.label(format!("Loss: {:.4}", progress.loss));
    ui.label(format!("Accuracy: {:.2}%", progress.accuracy * 100.0));
}
