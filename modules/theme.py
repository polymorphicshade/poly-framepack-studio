"""
The "poly" look: the same palette, type and shapes as the poly-wan web UI, as a Gradio theme.

Light and dark follow the browser's colour scheme, like poly-wan. Colours are set on Gradio's
theme variables, so the CSS below refers to those (--body-background-fill, --color-accent, ...)
and switches with the theme instead of hard-coding either mode.
"""

import gradio as gr

THEME_NAME = "poly"

# poly-wan's tokens. Light: bg #f6f6f4, panel #fff, panel-2 #efefec, border #dcdcd6,
# muted #6b6b66. Dark: bg #151517, panel #1e1e21, panel-2 #26262a, border #34343a, muted #9a9a94.
NEUTRAL = gr.themes.Color(
    name="poly-neutral",
    c50="#f6f6f4", c100="#efefec", c200="#dcdcd6", c300="#c4c4bd", c400="#9a9a94", c500="#6b6b66",
    c600="#4f4f4b", c700="#34343a", c800="#26262a", c900="#1e1e21", c950="#151517",
)
# Accent #5b4bdb in light mode and #8b7dff in dark mode.
ACCENT = gr.themes.Color(
    name="poly-accent",
    c50="#f1effd", c100="#e4e0fb", c200="#cbc4f7", c300="#ada2f2", c400="#8b7dff", c500="#6f5fe8",
    c600="#5b4bdb", c700="#4b3dbb", c800="#3d3297", c900="#302874", c950="#1f1a4b",
)

TEXT = "#1d1d1b"
TEXT_DARK = "#ececea"
DANGER = "#c0392b"
OK = "#2e8b57"
WARN = "#c98a12"


def poly_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=ACCENT,
        secondary_hue=ACCENT,
        neutral_hue=NEUTRAL,
        font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
        text_size=gr.themes.Size(xxs="10px", xs="11px", sm="12.5px", md="14px", lg="15px", xl="17px", xxl="22px"),
        radius_size=gr.themes.Size(xxs="2px", xs="3px", sm="4px", md="6px", lg="8px", xl="12px", xxl="16px"),
    )
    return theme.set(
        # surfaces
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        body_text_color=TEXT,
        body_text_color_dark=TEXT_DARK,
        body_text_color_subdued="*neutral_500",
        body_text_color_subdued_dark="*neutral_400",
        background_fill_primary="white",
        background_fill_primary_dark="*neutral_900",
        background_fill_secondary="*neutral_100",
        background_fill_secondary_dark="*neutral_800",
        border_color_primary="*neutral_200",
        border_color_primary_dark="*neutral_700",
        border_color_accent="*primary_600",
        border_color_accent_dark="*primary_400",
        color_accent="*primary_600",
        color_accent_soft="*primary_50",
        color_accent_soft_dark="*neutral_800",
        link_text_color="*primary_600",
        link_text_color_dark="*primary_400",
        link_text_color_hover="*primary_700",
        link_text_color_hover_dark="*primary_300",
        link_text_color_visited="*primary_600",
        link_text_color_visited_dark="*primary_400",
        shadow_drop="none",
        shadow_drop_lg="0 4px 16px rgba(0,0,0,.12)",
        layout_gap="12px",
        # blocks and labels
        block_background_fill="white",
        block_background_fill_dark="*neutral_900",
        block_border_color="*neutral_200",
        block_border_color_dark="*neutral_700",
        block_border_width="1px",
        block_radius="*radius_lg",
        block_shadow="none",
        block_label_background_fill="*neutral_100",
        block_label_background_fill_dark="*neutral_800",
        block_label_text_color="*neutral_500",
        block_label_text_color_dark="*neutral_400",
        block_label_text_size="*text_sm",
        block_label_text_weight="600",
        block_title_text_color="*neutral_500",
        block_title_text_color_dark="*neutral_400",
        block_title_text_size="*text_sm",
        block_title_text_weight="600",
        block_info_text_color="*neutral_500",
        block_info_text_color_dark="*neutral_400",
        panel_background_fill="white",
        panel_background_fill_dark="*neutral_900",
        panel_border_color="*neutral_200",
        panel_border_color_dark="*neutral_700",
        # inputs
        input_background_fill="white",
        input_background_fill_dark="*neutral_900",
        input_border_color="*neutral_200",
        input_border_color_dark="*neutral_700",
        input_border_color_focus="*primary_600",
        input_border_color_focus_dark="*primary_400",
        input_radius="*radius_md",
        input_shadow="none",
        input_shadow_focus="none",
        checkbox_background_color_selected="*primary_600",
        checkbox_background_color_selected_dark="*primary_400",
        checkbox_border_color_selected="*primary_600",
        checkbox_border_color_selected_dark="*primary_400",
        checkbox_label_background_fill="*neutral_100",
        checkbox_label_background_fill_dark="*neutral_800",
        checkbox_label_background_fill_selected="*primary_50",
        checkbox_label_background_fill_selected_dark="*neutral_700",
        checkbox_label_border_color="*neutral_200",
        checkbox_label_border_color_dark="*neutral_700",
        checkbox_label_shadow="none",
        slider_color="*primary_600",
        slider_color_dark="*primary_400",
        loader_color="*primary_600",
        loader_color_dark="*primary_400",
        # buttons: accent primary, panel-2 secondary, red stop
        button_large_radius="*radius_md",
        button_small_radius="*radius_md",
        button_medium_radius="*radius_md",
        button_large_text_weight="600",
        button_medium_text_weight="500",
        button_primary_shadow="none",
        button_primary_shadow_hover="none",
        button_primary_shadow_active="none",
        button_secondary_shadow="none",
        button_secondary_shadow_hover="none",
        button_secondary_shadow_active="none",
        button_transition="border-color .12s, background-color .12s",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_dark="*primary_400",
        button_primary_background_fill_hover="*primary_700",
        button_primary_background_fill_hover_dark="*primary_300",
        button_primary_border_color="*primary_600",
        button_primary_border_color_dark="*primary_400",
        button_primary_text_color="white",
        button_primary_text_color_dark="#111111",
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_dark="*neutral_800",
        button_secondary_background_fill_hover="*neutral_100",
        button_secondary_background_fill_hover_dark="*neutral_800",
        button_secondary_border_color="*neutral_200",
        button_secondary_border_color_dark="*neutral_700",
        button_secondary_border_color_hover="*neutral_500",
        button_secondary_border_color_hover_dark="*neutral_400",
        button_secondary_text_color=TEXT,
        button_secondary_text_color_dark=TEXT_DARK,
        button_cancel_background_fill=DANGER,
        button_cancel_background_fill_dark=DANGER,
        button_cancel_background_fill_hover="#a93226",
        button_cancel_background_fill_hover_dark="#a93226",
        button_cancel_border_color=DANGER,
        button_cancel_border_color_dark=DANGER,
        button_cancel_text_color="white",
        button_cancel_text_color_dark="white",
        # tables (queue)
        table_border_color="*neutral_200",
        table_border_color_dark="*neutral_700",
        table_even_background_fill="white",
        table_even_background_fill_dark="*neutral_900",
        table_odd_background_fill="*neutral_50",
        table_odd_background_fill_dark="*neutral_950",
        table_row_focus="*primary_50",
        table_row_focus_dark="*neutral_800",
        error_background_fill="#fbecea",
        error_background_fill_dark="#3a1f1c",
        error_border_color=DANGER,
        error_border_color_dark=DANGER,
        error_text_color=DANGER,
        error_text_color_dark="#e8806f",
    )


def resolve_theme(name):
    """The theme to pass to gr.Blocks for a theme name from the settings."""
    if not name or name in (THEME_NAME, "default"):
        return poly_theme()
    return name


# Layout and component styling on top of the theme: poly-wan's top bar, folder tabs, segmented
# mode buttons, flat cards and thin progress bar.
CSS = """
.gradio-container { padding-top: 44px !important; }
.gradio-container > main { padding: 0 !important; }
footer { opacity: .7; }

/* top bar */
#fixed-toolbar {
  position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
  height: 44px; padding: 0 12px; gap: 8px !important;
  display: flex; align-items: center; flex-wrap: nowrap !important;
  background: var(--background-fill-primary);
  border-bottom: 1px solid var(--border-color-primary);
  color: var(--body-text-color);
}
#fixed-toolbar > .column { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
#fixed-toolbar > .column.toolbar-queue { flex: 1 1 0 !important; }
#fixed-toolbar .toolbar-stat-textbox { min-width: 0 !important; }
#toolbar-ram-col { width: 170px !important; }
#toolbar-vram-col { width: 140px !important; }
#toolbar-gpu-col { width: 120px !important; }
#fixed-toolbar .block, #fixed-toolbar .form, #fixed-toolbar .prose { background: none !important; border: none !important; padding: 0 !important; box-shadow: none !important; }
#fixed-toolbar .html-container { padding: 0 !important; }
.toolbar-brand { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
.toolbar-title { margin: 0 !important; font-size: 14px !important; font-weight: 700 !important; color: var(--body-text-color) !important; }
.toolbar-version {
  margin: 0 !important; font-size: 11px; line-height: 1.5; padding: 1px 6px;
  border: 1px solid var(--border-color-primary); border-radius: 999px; color: var(--body-text-color-subdued);
}
.toolbar-text { margin: 0 !important; font-size: 13px; color: var(--body-text-color-subdued); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.toolbar-stat-textbox textarea, .toolbar-stat-textbox input {
  height: 28px !important; min-height: 0 !important; padding: 3px 12px !important; resize: none !important;
  font-size: 12.5px !important; line-height: 20px !important; text-align: center; white-space: nowrap; overflow: hidden;
  background: var(--background-fill-secondary) !important; color: var(--body-text-color) !important;
  border: 1px solid var(--border-color-primary) !important; border-radius: 999px !important; box-shadow: none !important;
}
@media (max-width: 900px) { .toolbar-text { display: none; } }
@media (max-width: 640px) { #toolbar-ram-col, #toolbar-gpu-col, .toolbar-version { display: none !important; } }

/* main tabs: poly-wan's folder tabs, on the panel colour under the top bar */
#main_tabs { gap: 0 !important; }
#main_tabs > .tab-wrapper {
  background: var(--background-fill-primary); border-bottom: 1px solid var(--border-color-primary);
  padding: 6px 12px 0; margin-bottom: 14px; height: auto;
}
#main_tabs > .tab-wrapper .tab-container { gap: 2px; height: auto; }
#main_tabs > .tab-wrapper button[role="tab"] {
  padding: 6px 14px; height: auto; border: 1px solid var(--border-color-primary); border-bottom: none;
  border-radius: 8px 8px 0 0; background: var(--background-fill-secondary); color: var(--body-text-color-subdued);
  font-size: 14px; font-weight: 500; margin-bottom: -1px;
}
#main_tabs > .tab-wrapper button[role="tab"].selected { background: var(--body-background-fill); color: var(--body-text-color); }
#main_tabs > .tab-wrapper button[role="tab"]::after { display: none !important; }
#main_tabs > .tabitem { padding: 0 14px 14px; border: none; background: none; }

/* nested tabs: poly-wan's segmented mode buttons */
#main_tabs .tabitem .tab-wrapper { border-bottom: none; height: auto; padding: 0; }
#main_tabs .tabitem .tab-container {
  display: flex; flex-wrap: wrap; gap: 1px; height: auto; background: var(--border-color-primary);
  border: 1px solid var(--border-color-primary); border-radius: 8px; overflow: hidden;
}
#main_tabs .tabitem .tab-container button[role="tab"] {
  flex: 1 1 auto; justify-content: center; text-align: center; height: auto; padding: 7px 8px;
  border: none; border-radius: 0; margin: 0;
  background: var(--background-fill-primary); color: var(--body-text-color);
}
#main_tabs .tabitem .tab-container button[role="tab"].selected { background: var(--color-accent); color: var(--button-primary-text-color); font-weight: 600; }
#main_tabs .tabitem .tab-container button[role="tab"]::after { display: none !important; }

/* generation type: segmented, like poly-wan's mode row (":not(.full)" skips Gradio's loading overlay) */
/* The block keeps the panel colour: Gradio's .form wrapper behind it is border-coloured. */
.poly-modes { padding: 0 !important; border: none !important; background: var(--background-fill-primary) !important; }
.form:has(> .poly-modes) { border: none !important; background: none !important; }
.poly-modes > [data-testid="block-info"] { background: none !important; padding: 0 0 6px !important; margin: 0 !important; }
.poly-modes > .wrap:not(.full) {
  display: flex !important; flex-wrap: wrap; gap: 1px !important; background: var(--border-color-primary);
  border: 1px solid var(--border-color-primary); border-radius: 8px; overflow: hidden;
}
.poly-modes > .wrap:not(.full) label {
  flex: 1 1 auto; justify-content: center; margin: 0 !important; padding: 7px 8px !important;
  border: none !important; border-radius: 0 !important; box-shadow: none !important;
  background: var(--background-fill-primary) !important; color: var(--body-text-color) !important; cursor: pointer;
}
.poly-modes > .wrap:not(.full) label.selected { background: var(--color-accent) !important; color: var(--button-primary-text-color) !important; font-weight: 600; }
.poly-modes > .wrap:not(.full) label input { display: none !important; }

/* control panel (left) and stage (right) */
.poly-controls {
  background: var(--background-fill-primary); border: 1px solid var(--border-color-primary);
  border-radius: 8px; padding: 14px !important;
}
.poly-stage { padding: 0 4px !important; }
.gradio-container .label-wrap { font-weight: 600; }
.gradio-container .label-wrap span { font-size: 14px; }
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 { font-weight: 650; letter-spacing: 0; }
.gradio-container .prose h4 { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--body-text-color-subdued); margin: 10px 0 2px; }
.control-group { border-top: 1px solid var(--border-color-primary) !important; border-bottom: 1px solid var(--border-color-primary) !important; margin: 12px 0; }

/* checkerboard behind images, as poly-wan's slots and results */
.contain-image .image-container, .contain-image .image-frame {
  background: repeating-conic-gradient(var(--background-fill-secondary) 0 25%, var(--background-fill-primary) 0 50%) 0 0 / 14px 14px;
}

/* status messages */
.poly-status { margin: 0; font-size: 13px; }
.poly-status.ok { color: #2e8b57; }
.poly-status.error { color: #c0392b; }
.poly-status.info { color: var(--color-accent); }
.poly-status.muted { color: var(--body-text-color-subdued); }
.poly-status.center { text-align: center; }
"""
