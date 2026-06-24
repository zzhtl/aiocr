use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;

const SYSTEM_CJK_FONT_NAME: &str = "system-cjk";
const BUNDLED_CJK_FONT_NAME: &str = "bundled-cjk";
const SEARCH_DEPTH_LIMIT: usize = 4;

/// 内置兜底中文字体（Droid Sans Fallback Full，Apache-2.0），保证无系统字体时中文也能显示。
const BUNDLED_CJK_FONT: &[u8] = include_bytes!("../assets/fallback-cjk.ttf");

pub fn configure_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // 内置兜底中文字体：始终注册为最后备选，保证中文必显（不再硬依赖系统字体）。
    fonts.font_data.insert(
        BUNDLED_CJK_FONT_NAME.to_string(),
        Arc::new(egui::FontData::from_static(BUNDLED_CJK_FONT)),
    );

    // 系统中文字体：找到则优先使用（字形更贴近平台、渲染更佳）。
    let system_font = load_system_cjk_font();
    if let Some((path, bytes)) = &system_font {
        fonts.font_data.insert(
            SYSTEM_CJK_FONT_NAME.to_string(),
            Arc::new(egui::FontData::from_owned(bytes.clone())),
        );
        tracing::info!("加载系统中文字体: {}", path.display());
    } else {
        tracing::info!("未找到系统中文字体，使用内置兜底中文字体");
    }

    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        if let Some(list) = fonts.families.get_mut(&family) {
            if system_font.is_some() {
                list.insert(0, SYSTEM_CJK_FONT_NAME.to_string());
            }
            list.push(BUNDLED_CJK_FONT_NAME.to_string());
        }
    }

    ctx.set_fonts(fonts);
}

pub fn load_clipboard_image() -> Result<image::DynamicImage, String> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|err| format!("初始化系统剪贴板失败: {err}"))?;
    let image = clipboard
        .get_image()
        .map_err(|err| format!("读取剪贴板图片失败: {err}"))?;

    let bytes = image.bytes.into_owned();
    let expected_len = image.width * image.height * 4;
    if bytes.len() != expected_len {
        return Err(format!(
            "剪贴板图片像素长度异常: got {}, expected {}",
            bytes.len(),
            expected_len
        ));
    }

    let rgba = image::RgbaImage::from_vec(image.width as u32, image.height as u32, bytes)
        .ok_or_else(|| "剪贴板图片像素解码失败".to_string())?;

    Ok(image::DynamicImage::ImageRgba8(rgba))
}

fn load_system_cjk_font() -> Option<(PathBuf, Vec<u8>)> {
    let candidates = system_cjk_font_candidates();
    let roots = font_search_roots();
    let paths = scan_font_roots(&roots, &candidates);

    candidates.into_iter().find_map(|candidate| {
        paths
            .get(&candidate.to_ascii_lowercase())
            .and_then(|path| std::fs::read(path).ok().map(|bytes| (path.clone(), bytes)))
    })
}

fn scan_font_roots(roots: &[PathBuf], candidates: &[&str]) -> HashMap<String, PathBuf> {
    let mut found = HashMap::new();
    let mut stack = Vec::new();
    let wanted = candidates
        .iter()
        .map(|name| name.to_ascii_lowercase())
        .collect::<Vec<_>>();

    for root in roots {
        if root.exists() {
            stack.push((root.clone(), 0usize));
        }
    }

    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };

            if file_type.is_dir() {
                if depth < SEARCH_DEPTH_LIMIT {
                    stack.push((path, depth + 1));
                }
                continue;
            }

            if !file_type.is_file() {
                continue;
            }

            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let file_name = file_name.to_ascii_lowercase();

            if wanted.iter().any(|candidate| candidate == &file_name) {
                found.entry(file_name).or_insert(path);
            }
        }
    }

    found
}

fn font_search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    #[cfg(target_os = "windows")]
    {
        if let Ok(windir) = std::env::var("WINDIR") {
            roots.push(PathBuf::from(windir).join("Fonts"));
        }
        roots.push(PathBuf::from(r"C:\Windows\Fonts"));
    }

    #[cfg(target_os = "macos")]
    {
        roots.push(PathBuf::from("/System/Library/Fonts"));
        roots.push(PathBuf::from("/Library/Fonts"));
        if let Some(home) = user_home_dir() {
            roots.push(home.join("Library/Fonts"));
        }
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        roots.push(PathBuf::from("/usr/share/fonts"));
        roots.push(PathBuf::from("/usr/local/share/fonts"));
        if let Some(home) = user_home_dir() {
            roots.push(home.join(".local/share/fonts"));
            roots.push(home.join(".fonts"));
        }
    }

    roots.retain(|root| root.exists());
    roots.sort();
    roots.dedup();
    roots
}

fn user_home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

fn system_cjk_font_candidates() -> Vec<&'static str> {
    let mut candidates = Vec::new();

    #[cfg(target_os = "windows")]
    {
        candidates.extend([
            "msyh.ttc",
            "msyh.ttf",
            "msyhbd.ttc",
            "simhei.ttf",
            "simsun.ttc",
            "Deng.ttf",
            "Dengb.ttf",
        ]);
    }

    #[cfg(target_os = "macos")]
    {
        candidates.extend([
            "PingFang.ttc",
            "Hiragino Sans GB.ttc",
            "Songti.ttc",
            "STHeiti Light.ttc",
            "Arial Unicode.ttf",
        ]);
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        candidates.extend([
            "NotoSansCJKsc-Regular.otf",
            "NotoSansSC-Regular.otf",
            "NotoSansCJK-Regular.ttc",
            "SourceHanSansSC-Regular.otf",
            "SourceHanSansCN-Regular.otf",
            "wqy-microhei.ttc",
            "WenQuanYi Micro Hei.ttf",
            "SarasaUiSC-Regular.ttf",
            "LXGWWenKai-Regular.ttf",
        ]);
    }

    candidates
}
