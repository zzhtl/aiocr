mod app;
mod native;
mod panels;
mod state;
mod theme;
mod widgets;
mod worker;

fn main() -> eframe::Result<()> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aiocr=info".into()),
        )
        .init();

    tracing::info!("AIOCR 启动");
    #[cfg(debug_assertions)]
    tracing::warn!(
        "当前为 debug 构建，本地 OCR 推理会明显偏慢且更占 CPU，建议使用 cargo run --release -p aiocr-gui"
    );

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("AIOCR - 图片文字识别"),
        ..Default::default()
    };

    eframe::run_native(
        "AIOCR",
        options,
        Box::new(|cc| Ok(Box::new(app::AiocrApp::new(cc)))),
    )
}
