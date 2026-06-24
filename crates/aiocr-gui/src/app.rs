use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use aiocr_core::build_layout_text;
use eframe::egui;

use crate::native;
use crate::panels::{
    control_panel, image_panel, model_hub_panel, result_panel, toolbar, training_panel,
};
use crate::state::{AppState, DownloadStatus, EngineBackend, TaskStatus};
use crate::theme;
use crate::worker::{DirectoryKind, EngineSpec, Worker, WorkerMessage};

const PASTE_IMAGE_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::V);
const PASTE_IMAGE_SHORTCUT_ALT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Insert);

/// AIOCR 桌面应用
pub struct AiocrApp {
    state: AppState,
    worker: Worker,
    receiver: mpsc::Receiver<WorkerMessage>,
    texture: Option<egui::TextureHandle>,
    active_tab: Tab,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Ocr,
    Training,
    ModelHub,
}

impl AiocrApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        native::configure_fonts(&cc.egui_ctx);

        let (sender, receiver) = mpsc::channel();
        let state = AppState::default();
        theme::install(&cc.egui_ctx, state.theme_mode);
        let worker = Worker::new(sender);
        worker.refresh_models(state.training.config.artifact_dir.clone());

        Self {
            state,
            worker,
            receiver,
            texture: None,
            active_tab: Tab::Ocr,
        }
    }

    /// 构建当前应激活的引擎规格
    fn current_engine_spec(&self) -> EngineSpec {
        match &self.state.active_backend {
            EngineBackend::BurnDefault => EngineSpec::Default,
            EngineBackend::LocalAi => {
                if let Some(dir) = &self.state.training.active_model_dir {
                    EngineSpec::LocalAi(dir.clone())
                } else {
                    EngineSpec::Default
                }
            }
            EngineBackend::Onnx => EngineSpec::Onnx {
                det_path: self.state.onnx.det_path.clone(),
                rec_path: self.state.onnx.rec_path.clone(),
                dict_path: self.state.onnx.dict_path.clone(),
            },
        }
    }

    /// 处理后台消息
    fn process_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.receiver.try_recv() {
            ctx.request_repaint();
            match msg {
                WorkerMessage::ImageLoaded(path, img) => {
                    self.set_image(ctx, img, Some(path), "图片加载完成");
                }
                WorkerMessage::OcrComplete(result) => {
                    self.state.spatial_text = self.state.image_data.as_ref().map(|image| {
                        build_layout_text(
                            &result.regions,
                            image.width() as f32,
                            image.height() as f32,
                        )
                    });
                    self.state.status_message = format!(
                        "识别完成: {} 个区域, {}ms",
                        result.regions.len(),
                        result.elapsed_ms
                    );
                    self.state.ocr_result = Some(result);
                    self.state.task_status = TaskStatus::Idle;
                }
                WorkerMessage::TrainingProgress(progress) => {
                    self.state.task_status = TaskStatus::Training;
                    self.state.status_message = format!(
                        "训练中: Epoch {}/{} Batch {}/{} loss={:.4} acc={:.1}%",
                        progress.epoch,
                        progress.total_epochs,
                        progress.batch,
                        progress.total_batches,
                        progress.loss,
                        progress.accuracy * 100.0,
                    );
                    self.state.training.progress = Some(progress);
                }
                WorkerMessage::TrainingComplete(summary) => {
                    self.state.task_status = TaskStatus::Idle;
                    self.state.training.progress = None;
                    self.state.training.selected_model_dir = Some(summary.model.model_dir.clone());
                    self.state.training.active_model_dir = Some(summary.model.model_dir.clone());
                    self.state.status_message =
                        format!("训练完成，已切换到模型: {}", summary.model.display_name());
                    self.worker
                        .refresh_models(self.state.training.config.artifact_dir.clone());
                }
                WorkerMessage::ModelsLoaded(models) => {
                    self.state.training.set_available_models(models);
                }
                WorkerMessage::DirectorySelected(kind, path) => {
                    self.handle_directory_selected(kind, path);
                }
                WorkerMessage::DownloadProgress {
                    component,
                    downloaded,
                    total,
                } => {
                    let mb = downloaded as f32 / 1_048_576.0;
                    self.state.status_message = format!("下载中: {component} ({mb:.1} MB)");
                    self.state.model_hub.download_status = DownloadStatus::Downloading {
                        component,
                        downloaded,
                        total,
                    };
                }
                WorkerMessage::DownloadComplete(save_dir) => {
                    self.state.model_hub.download_status = DownloadStatus::Complete;
                    self.state.onnx.load_from_dir(save_dir);
                    self.worker.invalidate_onnx_cache();
                    if self.state.onnx.is_usable() {
                        self.state.active_backend = EngineBackend::Onnx;
                    }
                    self.state.status_message = format!("模型下载完成，已切换到外部 ONNX 推理引擎");
                }
                WorkerMessage::Error(msg) => {
                    if msg.contains("训练已取消") {
                        self.state.task_status = TaskStatus::Idle;
                        self.state.training.progress = None;
                        self.state.status_message = "训练已取消".to_string();
                    } else if msg.contains("模型下载失败") {
                        self.state.model_hub.download_status = DownloadStatus::Error(msg.clone());
                        self.state.status_message = msg;
                    } else {
                        self.state.task_status = TaskStatus::Error(msg.clone());
                        self.state.status_message = msg;
                    }
                }
            }
        }
    }

    fn handle_directory_selected(&mut self, kind: DirectoryKind, path: PathBuf) {
        match kind {
            DirectoryKind::Dataset => {
                self.state.training.config.dataset_path = path.clone();
                self.state.status_message = format!("数据集目录已更新: {}", path.display());
            }
            DirectoryKind::Artifact => {
                self.state.training.config.artifact_dir = path.clone();
                self.state.status_message = format!("产物目录已更新: {}", path.display());
                self.worker.refresh_models(path);
            }
            DirectoryKind::OnnxModel => {
                self.state.onnx.load_from_dir(path.clone());
                self.worker.invalidate_onnx_cache();
                self.state.status_message = format!("ONNX 模型目录已设置: {}", path.display());
                if self.state.onnx.is_usable() {
                    self.state.active_backend = EngineBackend::Onnx;
                    self.state.status_message =
                        format!("已切换到外部 ONNX 模型: {}", self.state.onnx.status());
                }
            }
            DirectoryKind::ModelHubSave => {
                self.state.model_hub.save_dir = path;
            }
        }
    }

    /// 更新图片纹理
    fn update_texture(&mut self, ctx: &egui::Context, img: &image::DynamicImage) {
        let rgba = img.to_rgba8();
        let size = [rgba.width() as usize, rgba.height() as usize];
        let pixels = rgba.as_flat_samples();
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
        self.texture = Some(ctx.load_texture("ocr-image", color_image, Default::default()));
    }

    fn set_image(
        &mut self,
        ctx: &egui::Context,
        img: image::DynamicImage,
        path: Option<PathBuf>,
        status: &str,
    ) {
        self.update_texture(ctx, &img);
        self.state.image_path = path;
        self.state.image_data = Some(Arc::new(img));
        self.state.ocr_result = None;
        self.state.spatial_text = None;
        self.state.task_status = TaskStatus::Idle;
        self.state.status_message = status.to_string();
    }

    fn start_ocr(&mut self) {
        if self.state.task_status.is_busy() {
            return;
        }

        if let Some(image) = self.state.image_data.clone() {
            self.state.task_status = TaskStatus::Recognizing;
            self.state.status_message = "识别中...".to_string();
            let spec = self.current_engine_spec();
            self.worker.run_ocr(image, spec, self.state.quality_preset);
        }
    }

    fn paste_image_from_clipboard(&mut self, ctx: &egui::Context, auto_recognize: bool) {
        if self.state.task_status.is_busy() {
            return;
        }

        match native::load_clipboard_image() {
            Ok(image) => {
                self.set_image(ctx, image, None, "已从剪贴板载入截图");
                if auto_recognize {
                    self.start_ocr();
                }
            }
            Err(err) => {
                self.state.task_status = TaskStatus::Error(err.clone());
                self.state.status_message = err;
            }
        }
    }

    fn handle_toolbar_action(&mut self, ctx: &egui::Context, action: toolbar::ToolbarAction) {
        match action {
            toolbar::ToolbarAction::OpenImage => {
                self.state.task_status = TaskStatus::Loading;
                self.state.status_message = "选择图片...".to_string();
                self.worker.open_image();
            }
            toolbar::ToolbarAction::PasteImage => {
                self.paste_image_from_clipboard(ctx, true);
            }
            toolbar::ToolbarAction::StartOcr => {
                self.start_ocr();
            }
            toolbar::ToolbarAction::ToggleTheme => {
                self.state.theme_mode = self.state.theme_mode.next();
                theme::install(ctx, self.state.theme_mode);
            }
            toolbar::ToolbarAction::None => {}
        }
    }

    fn process_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped_paths = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect::<Vec<_>>()
        });

        if let Some(path) = dropped_paths.into_iter().next() {
            self.state.task_status = TaskStatus::Loading;
            self.state.status_message = format!("加载拖拽图片: {}", path.display());
            self.worker.load_image_path(path);
        }
    }

    fn process_shortcuts(&mut self, ctx: &egui::Context) {
        if self.state.task_status.is_busy() {
            return;
        }

        let pressed = ctx.input_mut(|input| {
            input.consume_shortcut(&PASTE_IMAGE_SHORTCUT)
                || input.consume_shortcut(&PASTE_IMAGE_SHORTCUT_ALT)
        });

        if pressed {
            self.paste_image_from_clipboard(ctx, true);
        }
    }

    fn handle_training_action(&mut self, action: training_panel::TrainingAction) {
        match action {
            training_panel::TrainingAction::None => {}
            training_panel::TrainingAction::PickDataset => {
                self.worker.pick_directory(
                    DirectoryKind::Dataset,
                    Some(self.state.training.config.dataset_path.clone()),
                );
            }
            training_panel::TrainingAction::PickArtifactDir => {
                self.worker.pick_directory(
                    DirectoryKind::Artifact,
                    Some(self.state.training.config.artifact_dir.clone()),
                );
            }
            training_panel::TrainingAction::StartTraining => {
                self.state.task_status = TaskStatus::Training;
                self.state.training.progress = None;
                self.state.status_message = "训练中...".to_string();
                let mut config = self.state.training.config.clone();
                config.base_model_dir = self.state.training.active_model_dir.clone();
                self.worker.train(config);
            }
            training_panel::TrainingAction::CancelTraining => {
                self.worker.cancel_training();
            }
            training_panel::TrainingAction::RefreshModels => {
                self.worker
                    .refresh_models(self.state.training.config.artifact_dir.clone());
            }
            training_panel::TrainingAction::UseSelectedModel => {
                if let Some(selected) = self.state.training.selected_model_dir.clone() {
                    let model_name = self
                        .state
                        .training
                        .selected_model()
                        .map(|model| model.display_name())
                        .unwrap_or_else(|| selected.display().to_string());
                    self.state.training.active_model_dir = Some(selected);
                    self.state.active_backend = EngineBackend::LocalAi;
                    self.state.status_message = format!("已切换识别模型: {model_name}");
                    self.state.ocr_result = None;
                    self.state.spatial_text = None;
                }
            }
        }
    }

    fn handle_control_action(&mut self, action: control_panel::ControlAction) {
        match action {
            control_panel::ControlAction::None => {}
            control_panel::ControlAction::SwitchBackend(backend) => {
                self.state.active_backend = backend;
            }
            control_panel::ControlAction::PickOnnxDir => {
                self.worker
                    .pick_directory(DirectoryKind::OnnxModel, self.state.onnx.model_dir.clone());
            }
            control_panel::ControlAction::SetPreset(preset) => {
                self.state.quality_preset = preset;
                self.state.status_message =
                    format!("识别质量已设为：{}，重新识别后生效", preset.display_name());
            }
        }
    }
}

impl eframe::App for AiocrApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages(ctx);
        self.process_shortcuts(ctx);
        self.process_dropped_files(ctx);

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_tab, Tab::Ocr, "OCR 识别");
                ui.selectable_value(&mut self.active_tab, Tab::Training, "模型训练");
                ui.selectable_value(&mut self.active_tab, Tab::ModelHub, "模型下载");
                ui.separator();

                let action = toolbar::show(ui, &self.state);
                self.handle_toolbar_action(ctx, action);
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.state.task_status.is_busy() {
                    ui.spinner();
                }
                ui.label(&self.state.status_message);
            });
        });

        match self.active_tab {
            Tab::Ocr => self.show_ocr_tab(ctx),
            Tab::Training => self.show_training_tab(ctx),
            Tab::ModelHub => self.show_model_hub_tab(ctx),
        }
    }
}

impl AiocrApp {
    fn show_ocr_tab(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("right_panel")
            .default_width(560.0)
            .min_width(420.0)
            .max_width(720.0)
            .show(ctx, |ui| {
                egui::TopBottomPanel::bottom("ocr_control_panel")
                    .resizable(false)
                    .default_height(200.0)
                    .min_height(180.0)
                    .show_inside(ui, |ui| {
                        let action = control_panel::show(ui, &mut self.state);
                        self.handle_control_action(action);
                    });

                result_panel::show(ui, &self.state);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            image_panel::show(ui, &self.state, self.texture.as_ref());
        });
    }

    fn show_training_tab(&mut self, ctx: &egui::Context) {
        let is_training = matches!(self.state.task_status, TaskStatus::Training);
        egui::CentralPanel::default().show(ctx, |ui| {
            let action = training_panel::show(ui, &mut self.state.training, is_training);
            self.handle_training_action(action);
        });
    }

    fn show_model_hub_tab(&mut self, ctx: &egui::Context) {
        let is_busy = self.state.task_status.is_busy();
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let action = model_hub_panel::show(ui, &mut self.state.model_hub, is_busy);
                self.handle_hub_action(action);
            });
        });
    }

    fn handle_hub_action(&mut self, action: model_hub_panel::HubAction) {
        match action {
            model_hub_panel::HubAction::None => {}
            model_hub_panel::HubAction::PickSaveDir => {
                self.worker.pick_directory(
                    DirectoryKind::ModelHubSave,
                    Some(self.state.model_hub.save_dir.clone()),
                );
            }
            model_hub_panel::HubAction::StartDownload => {
                let hub = &self.state.model_hub;
                let dict_url = if hub.dict_url.trim().is_empty() {
                    None
                } else {
                    Some(hub.dict_url.clone())
                };
                self.state.status_message = "开始下载模型文件...".to_string();
                self.worker.download_models(
                    hub.det_url.clone(),
                    hub.rec_url.clone(),
                    dict_url,
                    hub.save_dir.clone(),
                );
                self.state.model_hub.download_status = DownloadStatus::Downloading {
                    component: "初始化...".to_string(),
                    downloaded: 0,
                    total: 0,
                };
            }
        }
    }
}
