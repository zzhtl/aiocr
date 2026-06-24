//! 统一的视觉主题：深浅双套配色 + 跟随系统，以及卡片/语义色辅助。
//!
//! 设计 token 集中在此，组件只引用语义色（accent/success/warning/error/card），
//! 不再散落硬编码颜色。

use eframe::egui::{self, Color32, CornerRadius, Margin, Stroke, Theme, ThemePreference, Visuals};

/// 圆角半径（设计 token）。
const CORNER_RADIUS: u8 = 8;
/// 卡片内边距。
const CARD_MARGIN: i8 = 10;

/// 主题模式：跟随系统 / 浅色 / 深色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemeMode {
    #[default]
    System,
    Light,
    Dark,
}

impl ThemeMode {
    fn preference(self) -> ThemePreference {
        match self {
            ThemeMode::System => ThemePreference::System,
            ThemeMode::Light => ThemePreference::Light,
            ThemeMode::Dark => ThemePreference::Dark,
        }
    }

    /// 标题栏切换按钮上的图标（用 BMP 符号，跨平台字体覆盖更稳）。
    pub fn icon(self) -> &'static str {
        match self {
            ThemeMode::System => "◐",
            ThemeMode::Light => "☀",
            ThemeMode::Dark => "☾",
        }
    }

    /// 中文标签。
    pub fn label(self) -> &'static str {
        match self {
            ThemeMode::System => "跟随系统",
            ThemeMode::Light => "浅色",
            ThemeMode::Dark => "深色",
        }
    }

    /// 循环切换：跟随系统 → 浅色 → 深色 → 跟随系统。
    pub fn next(self) -> Self {
        match self {
            ThemeMode::System => ThemeMode::Light,
            ThemeMode::Light => ThemeMode::Dark,
            ThemeMode::Dark => ThemeMode::System,
        }
    }
}

/// 安装两套自定义 Visuals + 统一间距，并设置当前主题偏好。
pub fn install(ctx: &egui::Context, mode: ThemeMode) {
    ctx.set_visuals_of(Theme::Dark, dark_visuals());
    ctx.set_visuals_of(Theme::Light, light_visuals());

    ctx.style_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(10.0, 6.0);
        style.spacing.menu_margin = Margin::same(8);
        style.spacing.window_margin = Margin::same(10);
    });

    ctx.set_theme(mode.preference());
}

fn dark_visuals() -> Visuals {
    let accent = Color32::from_rgb(122, 162, 247);
    let mut v = Visuals::dark();
    v.panel_fill = Color32::from_rgb(26, 27, 38);
    v.window_fill = Color32::from_rgb(31, 35, 53);
    v.extreme_bg_color = Color32::from_rgb(22, 22, 30);
    v.faint_bg_color = Color32::from_rgb(36, 40, 59);
    v.hyperlink_color = accent;
    v.warn_fg_color = Color32::from_rgb(224, 175, 104);
    v.error_fg_color = Color32::from_rgb(247, 118, 142);
    tune_widgets(
        &mut v,
        accent,
        Color32::from_rgb(41, 46, 66),
        Color32::from_rgb(51, 58, 82),
    );
    v
}

fn light_visuals() -> Visuals {
    let accent = Color32::from_rgb(59, 109, 240);
    let mut v = Visuals::light();
    v.panel_fill = Color32::from_rgb(246, 247, 249);
    v.window_fill = Color32::from_rgb(255, 255, 255);
    v.extreme_bg_color = Color32::from_rgb(255, 255, 255);
    v.faint_bg_color = Color32::from_rgb(236, 238, 242);
    v.hyperlink_color = accent;
    v.warn_fg_color = Color32::from_rgb(180, 120, 20);
    v.error_fg_color = Color32::from_rgb(200, 60, 60);
    tune_widgets(
        &mut v,
        accent,
        Color32::from_rgb(233, 235, 240),
        Color32::from_rgb(223, 227, 234),
    );
    v
}

/// 统一圆角，并给按钮等控件一套柔和的填充层级；选中态用 accent。
fn tune_widgets(v: &mut Visuals, accent: Color32, inactive_fill: Color32, hovered_fill: Color32) {
    let radius = CornerRadius::same(CORNER_RADIUS);
    v.window_corner_radius = CornerRadius::same(CORNER_RADIUS + 2);

    for widget in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        widget.corner_radius = radius;
    }

    v.widgets.inactive.weak_bg_fill = inactive_fill;
    v.widgets.inactive.bg_fill = inactive_fill;
    v.widgets.hovered.weak_bg_fill = hovered_fill;
    v.widgets.hovered.bg_fill = hovered_fill;

    v.selection.bg_fill = accent.gamma_multiply(0.40);
    v.selection.stroke = Stroke::new(1.0, accent);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, accent.gamma_multiply(0.6));
}

/// 卡片填充色（介于面板与控件之间，营造层级）。
pub fn card_fill(visuals: &Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_rgb(33, 36, 52)
    } else {
        Color32::from_rgb(255, 255, 255)
    }
}

/// 统一样式的卡片容器。
pub fn card_frame(visuals: &Visuals) -> egui::Frame {
    egui::Frame::new()
        .fill(card_fill(visuals))
        .inner_margin(Margin::same(CARD_MARGIN))
        .corner_radius(CornerRadius::same(CORNER_RADIUS))
        .stroke(Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color))
}

/// 强调色（按钮高亮、选中）。
pub fn accent(visuals: &Visuals) -> Color32 {
    visuals.hyperlink_color
}

/// 成功态绿色（egui 无内置语义绿，按深浅给两套）。
pub fn success(visuals: &Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_rgb(158, 206, 106)
    } else {
        Color32::from_rgb(46, 160, 90)
    }
}

/// 警告态（沿用 egui 语义色）。
pub fn warning(visuals: &Visuals) -> Color32 {
    visuals.warn_fg_color
}

/// 错误态（沿用 egui 语义色）。
pub fn error(visuals: &Visuals) -> Color32 {
    visuals.error_fg_color
}

/// 按识别置信度给出的徽章颜色：高=绿、中=黄、低=红。
pub fn confidence_color(visuals: &Visuals, confidence: f32) -> Color32 {
    if confidence >= 0.85 {
        success(visuals)
    } else if confidence >= 0.6 {
        warning(visuals)
    } else {
        error(visuals)
    }
}
