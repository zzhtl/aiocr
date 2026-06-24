use egui::Ui;

use crate::state::{DownloadStatus, ModelHubState};

/// 模型下载中心面板动作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubAction {
    None,
    PickSaveDir,
    StartDownload,
}

/// 模型下载中心面板。
pub fn show(ui: &mut Ui, hub: &mut ModelHubState, is_busy: bool) -> HubAction {
    let mut action = HubAction::None;

    ui.heading("模型下载中心");
    ui.separator();

    ui.label(
        egui::RichText::new(
            "从网络下载 PP-OCRv5 Server ONNX 模型文件（det.onnx + rec.onnx），\
             默认已预填桌面版中文模型链接，下载完成后会自动配置为外部 ONNX 推理引擎。",
        )
        .small()
        .weak(),
    );

    ui.add_space(8.0);

    show_url_fields(ui, hub);

    ui.add_space(6.0);

    show_save_dir(ui, hub, is_busy, &mut action);

    ui.add_space(8.0);

    show_download_button(ui, hub, is_busy, &mut action);

    ui.add_space(6.0);

    show_download_status(ui, hub);

    action
}

fn show_url_fields(ui: &mut Ui, hub: &mut ModelHubState) {
    ui.label("检测模型 URL（det.onnx）:");
    ui.add(
        egui::TextEdit::singleline(&mut hub.det_url)
            .desired_width(f32::INFINITY)
            .hint_text("默认已填入 PP-OCRv5 Server det.onnx 链接"),
    );

    ui.add_space(4.0);

    ui.label("识别模型 URL（rec.onnx）:");
    ui.add(
        egui::TextEdit::singleline(&mut hub.rec_url)
            .desired_width(f32::INFINITY)
            .hint_text("默认已填入 PP-OCRv5 Server rec.onnx 链接"),
    );

    ui.add_space(4.0);

    ui.label("字典文件 URL（ppocr_keys_v1.txt，可选）:");
    ui.add(
        egui::TextEdit::singleline(&mut hub.dict_url)
            .desired_width(f32::INFINITY)
            .hint_text("可选，粘贴字典文件链接（留空则使用内置字典）"),
    );

    ui.add_space(2.0);
    ui.label(
        egui::RichText::new(
            "提示：默认链接指向 PP-OCRv5 Server 中文模型；也可从 Hugging Face / GitHub Releases 获取其他 ONNX，或将官方 Paddle Inference 模型自行导出。",
        )
        .small()
        .weak(),
    );
}

fn show_save_dir(ui: &mut Ui, hub: &mut ModelHubState, is_busy: bool, action: &mut HubAction) {
    ui.label("模型保存目录:");
    ui.horizontal(|ui| {
        let mut path_str = hub.save_dir.display().to_string();
        ui.add_enabled(
            false,
            egui::TextEdit::singleline(&mut path_str).desired_width(400.0),
        );
        let downloading = hub.download_status.is_busy();
        if ui
            .add_enabled(!is_busy && !downloading, egui::Button::new("选择"))
            .clicked()
        {
            *action = HubAction::PickSaveDir;
        }
    });
}

fn show_download_button(
    ui: &mut Ui,
    hub: &mut ModelHubState,
    is_busy: bool,
    action: &mut HubAction,
) {
    let downloading = hub.download_status.is_busy();
    let has_urls = !hub.det_url.trim().is_empty() && !hub.rec_url.trim().is_empty();
    let can_download = has_urls && !downloading && !is_busy;

    ui.horizontal(|ui| {
        if ui
            .add_enabled(can_download, egui::Button::new("⬇ 开始下载"))
            .clicked()
        {
            *action = HubAction::StartDownload;
        }

        if !has_urls && !downloading {
            ui.label(
                egui::RichText::new("请先填写检测和识别模型 URL")
                    .small()
                    .weak(),
            );
        }
    });
}

fn show_download_status(ui: &mut Ui, hub: &ModelHubState) {
    match &hub.download_status {
        DownloadStatus::Idle => {}
        DownloadStatus::Downloading {
            component,
            downloaded,
            total,
        } => {
            ui.separator();
            ui.label(format!("正在下载: {component}"));
            if *total > 0 {
                let ratio = *downloaded as f32 / *total as f32;
                let mb_done = *downloaded as f32 / 1_048_576.0;
                let mb_total = *total as f32 / 1_048_576.0;
                ui.add(egui::ProgressBar::new(ratio).text(format!(
                    "{:.1}/{:.1} MB ({:.0}%)",
                    mb_done,
                    mb_total,
                    ratio * 100.0
                )));
            } else {
                let kb_done = *downloaded / 1024;
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .animate(true)
                        .text(format!("{kb_done} KB")),
                );
            }
        }
        DownloadStatus::Complete => {
            ui.separator();
            ui.label(
                egui::RichText::new("✓ 下载完成，已自动切换到外部 ONNX 推理引擎")
                    .color(crate::theme::success(ui.visuals())),
            );
        }
        DownloadStatus::Error(err) => {
            ui.separator();
            ui.label(
                egui::RichText::new(format!("✗ 下载失败: {err}"))
                    .color(crate::theme::error(ui.visuals())),
            );
        }
    }
}
