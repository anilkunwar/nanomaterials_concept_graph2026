"""
Streamlit app: W_k-weighted bar chart with full customization
================================================================
Parallel app for the **Q3 CS-sigmas** dataset (q3cs-sigmas-qdwa-weights.csv),
in which Cu@Ag Core-Shell is the principal (highest) bar:
    - Cu@Ag Core-Shell    W_k = 0.6806   Raw = 5.43
    - Electrodeposited nt-Cu  W_k = 0.2495   Raw = 1.83
    - Defect-Engineered Ag   W_k = 0.0699   Raw = 0.33

Bars are colored by the W_k colormap (right-hand colorbar), bar height
= "Raw" evidence value, with publication-grade typography, spines, ticks, etc.

Run it with:
    streamlit run bar_chart_app_q3cs.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager

# ----------------------------------------------------------------------------- #
#  Page config
# ----------------------------------------------------------------------------- #
st.set_page_config(
    page_title="W_k Bar Chart Studio — Q3 CS-sigmas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 W_k-weighted Bar Chart Studio — Q3 CS-sigmas")
st.markdown(
    "Publication-grade bar chart whose bar colors are driven by the **W_k** "
    "weight column (right-hand colorbar). This instance uses the **Q3 CS-sigmas** "
    "dataset, where **Cu@Ag Core-Shell is the principal bar** "
    "(W_k = 0.6806, Raw = 5.43). Every visual property can be tweaked in the sidebar."
)

# ----------------------------------------------------------------------------- #
#  Helpers
# ----------------------------------------------------------------------------- #
DEFAULT_CSV = Path(__file__).with_name("q3cs-sigmas-qdwa-weights.csv")

FALLBACK_DF = pd.DataFrame(
    {
        "Rank": [1, 2, 3],
        "Category": ["Cu@Ag Core-Shell", "Electrodeposited nt-Cu", "Defect-Engineered Ag"],
        "W_k":   [0.6806, 0.2495, 0.0699],
        "Raw":   [5.43, 1.83, 0.33],
    }
)


@st.cache_data
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def list_font_families() -> list[str]:
    """Return a sorted list of all font families matplotlib knows about."""
    return sorted({f.name for f in font_manager.fontManager.ttflist})


def all_colormaps() -> list[str]:
    """Sorted list of registered matplotlib colormaps."""
    return sorted(mpl.colormaps())


# ----------------------------------------------------------------------------- #
#  1. Data input  ----------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("1️⃣ Data input")
src = st.sidebar.radio(
    "Data source",
    ["Default CSV (bundled)", "Upload CSV", "Manual entry"],
    index=0,
    help="Use the bundled example data, upload your own CSV, or type values by hand.",
)

if src == "Default CSV (bundled)":
    df = load_csv(DEFAULT_CSV) if DEFAULT_CSV.exists() else FALLBACK_DF.copy()

elif src == "Upload CSV":
    up = st.sidebar.file_uploader("Choose a CSV file", type=["csv", "tsv"])
    if up is not None:
        try:
            df = pd.read_csv(up, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(up)
    else:
        df = FALLBACK_DF.copy()
        st.sidebar.info("No file uploaded yet — showing the bundled example data.")
else:
    st.sidebar.markdown("Type rows as `Category, W_k, Raw` (one row per line).")
    raw_txt = st.sidebar.text_area(
        "Manual data",
        "Cu@Ag Core-Shell, 0.6806, 5.43\nElectrodeposited nt-Cu, 0.2495, 1.83\nDefect-Engineered Ag, 0.0699, 0.33",
        height=120,
    )
    rows = []
    for line in raw_txt.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                rows.append({"Category": parts[0], "W_k": float(parts[1]), "Raw": float(parts[2])})
            except ValueError:
                pass
    df = pd.DataFrame(rows) if rows else FALLBACK_DF.copy()

st.subheader("Data preview")
st.dataframe(df, use_container_width=True, hide_index=True)

# Column selectors
cols = list(df.columns)
def_idx_cat = cols.index("Category") if "Category" in cols else 0
def_idx_val = cols.index("Raw") if "Raw" in cols else len(cols) - 1
def_idx_w  = cols.index("W_k")   if "W_k"   in cols else (1 if len(cols) > 1 else 0)

cat_col   = st.sidebar.selectbox("Category column (x-axis)", cols, index=def_idx_cat)
value_col = st.sidebar.selectbox("Value column (bar height)", cols, index=def_idx_val)
weight_col = st.sidebar.selectbox("Weight column (drives color)", cols, index=def_idx_w)

# ----------------------------------------------------------------------------- #
#  Bar ordering  ----------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.markdown("**Bar ordering**")
order_options = [
    "Original CSV order (Rank)",
    "By weight (highest first)  \u2014 default",
    "By weight (lowest first)",
    "By value (highest first)",
    "By value (lowest first)",
    "Category A \u2192 Z (alphabetical)",
    "Category Z \u2192 A (reverse alphabetical)",
    "Custom order (specify below)",
]
order_mode = st.sidebar.selectbox("Bar order", order_options, index=1,
    help="Reorder the bars along the x-axis. The default sorts by W_k descending "
         "(highest-weight category first), matching the reference figure.")

_custom_order_str = ""
if order_mode == "Custom order (specify below)":
    _custom_order_str = st.sidebar.text_area(
        "Custom category order (one per line, or comma-separated)",
        value="\n".join(df[cat_col].astype(str).tolist()),
        help="List the categories in the order you want them to appear on the x-axis. "
             "You can also paste a comma-separated list.",
        height=120,
    )

def _apply_order(d: pd.DataFrame, mode: str, custom: str) -> pd.DataFrame:
    """Return a copy of `d` reordered according to `mode`."""
    if mode == "Original CSV order (Rank)":
        return d.reset_index(drop=True)
    if mode == "By weight (highest first)  \u2014 default":
        return d.sort_values(weight_col, ascending=False, kind="stable").reset_index(drop=True)
    if mode == "By weight (lowest first)":
        return d.sort_values(weight_col, ascending=True, kind="stable").reset_index(drop=True)
    if mode == "By value (highest first)":
        return d.sort_values(value_col, ascending=False, kind="stable").reset_index(drop=True)
    if mode == "By value (lowest first)":
        return d.sort_values(value_col, ascending=True, kind="stable").reset_index(drop=True)
    if mode == "Category A \u2192 Z (alphabetical)":
        return d.sort_values(cat_col, ascending=True, kind="stable").reset_index(drop=True)
    if mode == "Category Z \u2192 A (reverse alphabetical)":
        return d.sort_values(cat_col, ascending=False, kind="stable").reset_index(drop=True)
    if mode == "Custom order (specify below)":
        sep = "\n" if "\n" in custom else ","
        ordered_names = [c.strip() for c in custom.split(sep) if c.strip()]
        name_to_row = {str(r[cat_col]): r for _, r in d.iterrows()}
        seen, ordered_rows = set(), []
        for name in ordered_names:
            if name in name_to_row and name not in seen:
                ordered_rows.append(name_to_row[name])
                seen.add(name)
        for _, r in d.iterrows():
            n = str(r[cat_col])
            if n not in seen:
                ordered_rows.append(r)
                seen.add(n)
        return pd.DataFrame(ordered_rows).reset_index(drop=True)
    return d.reset_index(drop=True)

df = _apply_order(df, order_mode, _custom_order_str)

# ----------------------------------------------------------------------------- #
#  2. Labels & title  -------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("2️⃣ Labels & title")
title_text  = st.sidebar.text_input("Chart title", "RAW EVIDENCE by CATEGORY (Q3 CS-sigmas)")
xlabel_txt = st.sidebar.text_input("X-axis label", "CATEGORY")
ylabel_txt = st.sidebar.text_input("Y-axis label", "RAW EVIDENCE")
cb_label   = st.sidebar.text_input("Colorbar label", r"$W_k$")
show_val_labels = st.sidebar.checkbox("Show value labels on bars", value=True)
val_label_fmt   = st.sidebar.text_input("Value label format", "{:.2f}")
val_label_off   = st.sidebar.slider("Value label vertical offset", 0.0, 1.0, 0.05, 0.01)

# ----------------------------------------------------------------------------- #
#  3. Fonts  ----------------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("3️⃣ Fonts")
font_families = list_font_families()
default_font  = "DejaVu Sans" if "DejaVu Sans" in font_families else font_families[0]
font_family   = st.sidebar.selectbox("Font family", font_families,
                                     index=font_families.index(default_font))
title_size    = st.sidebar.slider("Title size", 6, 36, 14)
label_size    = st.sidebar.slider("Axis-label size", 6, 36, 12)
tick_size     = st.sidebar.slider("Tick-label size", 4, 28, 10)
cb_size       = st.sidebar.slider("Colorbar-label size", 4, 28, 11)
title_weight   = st.sidebar.selectbox("Title weight",
                                      ["normal", "bold", "light", "semibold"], index=1)

# ----------------------------------------------------------------------------- #
#  4. Colormap  ------------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("4️⃣ Colormap")
cmaps = all_colormaps()
cmap_name = st.sidebar.selectbox("Colormap", cmaps, index=cmaps.index("viridis"))
cmap_reverse = st.sidebar.checkbox("Reverse colormap", value=False)
cb_norm_mode = st.sidebar.radio(
    "Colorbar normalization",
    ["Auto (min–max)", "Manual range"],
    index=0,
    horizontal=True,
)
if cb_norm_mode == "Manual range":
    cb_vmin = st.sidebar.number_input("Colorbar min", value=0.0, format="%.4f")
    cb_vmax = st.sidebar.number_input("Colorbar max", value=1.0, format="%.4f")
else:
    cb_vmin = cb_vmax = None

# ----------------------------------------------------------------------------- #
#  5. Bar / box properties  ------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("5️⃣ Bar / box width")
bar_width     = st.sidebar.slider("Bar width", 0.05, 1.0, 0.6, 0.05)
edge_color    = st.sidebar.color_picker("Bar edge color", "#000000")
edge_width    = st.sidebar.slider("Bar edge thickness", 0.0, 5.0, 0.8, 0.1)
bar_alpha     = st.sidebar.slider("Bar alpha", 0.1, 1.0, 1.0, 0.05)
bar_hatch     = st.sidebar.selectbox("Bar hatch",
                                     ["", "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"],
                                     index=0)

# ----------------------------------------------------------------------------- #
#  6. Axes & ticks  --------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("6️⃣ Axes & ticks")
axes_linewidth = st.sidebar.slider("Axes / spine linewidth", 0.2, 6.0, 1.5, 0.1)
tick_width     = st.sidebar.slider("Tick width", 0.2, 6.0, 1.2, 0.1)
tick_length    = st.sidebar.slider("Tick length", 0.0, 25.0, 5.0, 0.5)
tick_dir       = st.sidebar.selectbox("Tick direction", ["in", "out", "inout"], index=0)
tick_top       = st.sidebar.checkbox("Top ticks", value=False)
tick_bottom    = st.sidebar.checkbox("Bottom ticks", value=True)
tick_left      = st.sidebar.checkbox("Left ticks", value=True)
tick_right     = st.sidebar.checkbox("Right ticks", value=False)
xtick_rot      = st.sidebar.slider("X-tick label rotation (°)", 0, 90, 0, 5)
ytick_rot      = st.sidebar.slider("Y-tick label rotation (°)", 0, 90, 0, 5)

# --- X-axis label font enhancement -----------------------------------------
st.sidebar.markdown("**X-tick label font enhancement**")
x_tick_size_override = st.sidebar.checkbox("Custom X-tick label size", value=False,
    help="Turn on to set a font size for the X-tick labels that is independent "
         "of the general tick-label size above.")
if x_tick_size_override:
    x_tick_size = st.sidebar.slider("X-tick label size", 4, 48, 14)
else:
    x_tick_size = None  # falls back to the global tick_size
x_tick_weight  = st.sidebar.selectbox("X-tick label weight",
                                    ["normal", "bold", "light", "semibold"], index=1)
x_tick_color   = st.sidebar.color_picker("X-tick label color", "#000000")
x_tick_ha      = st.sidebar.selectbox("X-tick label horizontal alignment",
                                    ["center", "right", "left"], index=0,
    help="'right' is useful when labels are rotated (e.g. 45°).")
x_tick_use_math = st.sidebar.checkbox("Render X-tick labels as LaTeX",
                                      value=False,
    help="Wrap each label in $...$ so math like $\\downarrow$-Ag or $\\perp$-Ag renders properly. Off by default so plain text labels like 'nt-Cu' stay upright.")

# ----------------------------------------------------------------------------- #
#  7. Spines  --------------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("7️⃣ Spines")
spine_visible = {
    "top":    st.sidebar.checkbox("Top spine",    value=False),
    "bottom": st.sidebar.checkbox("Bottom spine", value=True),
    "left":   st.sidebar.checkbox("Left spine",   value=True),
    "right":  st.sidebar.checkbox("Right spine",  value=False),
}
spine_pos_mode = st.sidebar.selectbox("Spine position mode",
                                      ["outward (default)", "zero", "axes (0-1)", "data"],
                                      index=0)
if spine_pos_mode == "outward (default)":
    spine_offset = st.sidebar.slider("Outward offset (pts)", -50.0, 50.0, 0.0, 1.0)
else:
    spine_offset = None

# ----------------------------------------------------------------------------- #
#  8. Figure  --------------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("8️⃣ Figure")
fig_w   = st.sidebar.slider("Figure width (in)", 3.0, 16.0, 6.5, 0.25)
fig_h   = st.sidebar.slider("Figure height (in)", 2.0, 14.0, 5.0, 0.25)
dpi     = st.sidebar.slider("DPI", 72, 600, 150, 10)
facecol = st.sidebar.color_picker("Figure face color", "#FFFFFF")
ax_facecol = st.sidebar.color_picker("Axes face color", "#FFFFFF")
y_auto  = st.sidebar.checkbox("Auto Y-limits", value=True)
if not y_auto:
    y_min = st.sidebar.number_input("Y min", value=0.0)
    y_max = st.sidebar.number_input("Y max", value=6.0)
else:
    y_min = y_max = None
show_grid = st.sidebar.checkbox("Show grid", value=False)
grid_axis = st.sidebar.selectbox("Grid axis", ["both", "x", "y"], index=0) if show_grid else "both"
grid_alpha = st.sidebar.slider("Grid alpha", 0.05, 1.0, 0.3, 0.05) if show_grid else 0.3

# ----------------------------------------------------------------------------- #
#  9. Colorbar layout  -----------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("9️⃣ Colorbar")
show_cb        = st.sidebar.checkbox("Show colorbar", value=True)
cb_orientation = st.sidebar.selectbox("Colorbar orientation", ["vertical", "horizontal"])
cb_shrink      = st.sidebar.slider("Colorbar shrink", 0.2, 2.0, 0.8, 0.05)
cb_pad         = st.sidebar.slider("Colorbar pad", 0.0, 1.0, 0.04, 0.01)
cb_aspect      = st.sidebar.slider("Colorbar aspect", 5, 60, 20, 1)
cb_outline_w   = st.sidebar.slider("Colorbar outline width", 0.0, 5.0, 0.5, 0.1)

# ----------------------------------------------------------------------------- #
#  Per-bar customization  (main panel, after data preview + all sidebar sections)
#  Each bar gets its own panel for:
#    - X-tick label text
#    - Value-label text override (e.g. "6.33 ± 0.2", "High", or empty=auto)
#    - Show / hide the value label on this specific bar
#    - Bar color override (bypass the W_k colormap for one bar)
#    - Bar alpha override (per-bar transparency)
# ----------------------------------------------------------------------------- #
_default_categories = df[cat_col].astype(str).tolist()
_default_values    = pd.to_numeric(df[value_col], errors="coerce").tolist()
_n_bars           = len(_default_categories)

st.subheader("🎨 Per-bar customization")
st.markdown(
    "Edit each bar individually — its x-tick label, the value label shown on top "
    "of it (text + visibility), and optional bar color / alpha overrides that "
    "bypass the W_k colormap. All settings persist across reruns."
)

edited_labels:        list[str]        = []  # x-tick label text per bar
val_label_texts:      list[str]        = []  # custom value label text per bar ("" = auto)
val_label_show:       list[bool]       = []  # show value label on this bar?
bar_color_overrides:  list[str | None] = []  # hex color or None (use W_k colormap)
bar_alpha_overrides:  list[float | None] = [] # alpha or None (use global)

# Initialize per-bar session_state defaults
for i, orig in enumerate(_default_categories):
    defaults = {
        f"xlabel_{i}::{orig}":            orig,
        f"vlabel_text_{i}::{orig}":       "",
        f"vlabel_show_{i}::{orig}":       True,
        f"bar_color_override_{i}::{orig}": False,
        f"bar_color_val_{i}::{orig}":     "#1f77b4",
        f"bar_alpha_override_{i}::{orig}": False,
        f"bar_alpha_val_{i}::{orig}":     1.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ----------------------------------------------------------------------------- #
#  Symbol palette  --------------------------------------------------------------
#  Clickable buttons that append LaTeX snippets to a chosen target field
#  (a bar's X-tick label or value-label text). All snippets are wrapped in
#  $...$ so they render correctly whether or not the global "Render X-tick
#  labels as LaTeX" toggle is on.
# ----------------------------------------------------------------------------- #
SYMBOL_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Geometry / crystallography": [
        ("\u22a5",   r"$\perp$"),        # \u22a5 = \u22a5 (inverted T / perpendicular)
        ("\u2225",   r"$\parallel$"),    # parallel
        ("\u2220",   r"$\angle$"),       # angle
        ("\u0394",   r"$\Delta$"),       # Delta (lattice / difference)
        ("\u2207",   r"$\nabla$"),       # nabla
        ("\u2248",   r"$\approx$"),      # approx
        ("\u2261",   r"$\equiv$"),       # identical
        ("\u00b0",   r"$^\circ$"),       # degree
    ],
    "Arrows": [
        ("\u2191", r"$\uparrow$"),
        ("\u2193", r"$\downarrow$"),
        ("\u2190", r"$\leftarrow$"),
        ("\u2192", r"$\rightarrow$"),
        ("\u2194", r"$\leftrightarrow$"),
        ("\u21c4", r"$\rightleftharpoons$"),
        ("\u21d2", r"$\Rightarrow$"),
        ("\u21d0", r"$\Leftarrow$"),
        ("\u21d4", r"$\Leftrightarrow$"),
    ],
    "Greek lowercase": [
        ("\u03b1", r"$\alpha$"), ("\u03b2", r"$\beta$"),
        ("\u03b3", r"$\gamma$"), ("\u03b4", r"$\delta$"),
        ("\u03b5", r"$\epsilon$"), ("\u03b6", r"$\zeta$"),
        ("\u03b7", r"$\eta$"), ("\u03b8", r"$\theta$"),
        ("\u03b9", r"$\iota$"), ("\u03ba", r"$\kappa$"),
        ("\u03bb", r"$\lambda$"), ("\u03bc", r"$\mu$"),
        ("\u03bd", r"$\nu$"), ("\u03be", r"$\xi$"),
        ("\u03c0", r"$\pi$"), ("\u03c1", r"$\rho$"),
        ("\u03c3", r"$\sigma$"), ("\u03c4", r"$\tau$"),
        ("\u03c5", r"$\upsilon$"), ("\u03c6", r"$\phi$"),
        ("\u03c7", r"$\chi$"), ("\u03c8", r"$\psi$"),
        ("\u03c9", r"$\omega$"),
    ],
    "Greek uppercase": [
        ("\u0393", r"$\Gamma$"), ("\u0394", r"$\Delta$"),
        ("\u0398", r"$\Theta$"), ("\u039b", r"$\Lambda$"),
        ("\u039e", r"$\Xi$"), ("\u03a0", r"$\Pi$"),
        ("\u03a3", r"$\Sigma$"), ("\u03a6", r"$\Phi$"),
        ("\u03a8", r"$\Psi$"), ("\u03a9", r"$\Omega$"),
    ],
    "Math relations / operators": [
        ("\u00b1", r"$\pm$"),
        ("\u2213", r"$\mp$"),
        ("\u00d7", r"$\times$"),
        ("\u00f7", r"$\div$"),
        ("\u22c5", r"$\cdot$"),
        ("\u2260", r"$\neq$"),
        ("\u2264", r"$\leq$"),
        ("\u2265", r"$\geq$"),
        ("\u221d", r"$\propto$"),
        ("\u221e", r"$\infty$"),
        ("\u2211", r"$\sum$"),
        ("\u222b", r"$\int$"),
        ("\u221a", r"$\sqrt{}$"),
    ],
    "Subscripts / superscripts": [
        ("\u00b2", r"$^2$"), ("\u00b3", r"$^3$"),
        ("\u2070", r"$^0$"), ("\u2071", r"$^1$"),
        ("\u207f", r"$^n$"), ("\u1d4f", r"$^k$"),
        ("\u2080", r"$_0$"), ("\u2081", r"$_1$"),
        ("\u2082", r"$_2$"), ("\u2096", r"$_k$"),
        ("\u2099", r"$_n$"), ("\u2098", r"$_m$"),
    ],
    "Misc / dingbats": [
        ("\u2605", r"$\star$"),
        ("\u2020", r"$\dagger$"),
        ("\u2021", r"$\ddagger$"),
        ("\u00a7", r"$\S$"),
        ("\u00b6", r"$\P$"),
        ("\u2113", r"$\ell$"),
        ("\u210f", r"$\hbar$"),
        ("\u00c5", r"$\AA$"),  # Angstrom
        ("\u00b5", r"$\mu$"),  # micro
    ],
}

st.subheader("\U0001f5ce Symbol palette")
st.markdown(
    "Pick a target field below, then click any symbol to **append** its LaTeX "
    "snippet to that field. All snippets are wrapped in `$...$` so they render "
    "correctly whether or not the sidebar's *Render X-tick labels as LaTeX* "
    "toggle is on. Example: clicking \u22a5 inserts `$\\perp$` which renders as "
    "an inverted-T (perpendicular symbol)."
)

# Build target options (bar index + field name + session_state key)
_target_options: list[tuple[str, str]] = []
_target_keys:    list[str] = []
for i, orig in enumerate(_default_categories):
    # X-tick label target
    _target_options.append((f"Bar {i+1} \u2014 X-tick label", f"xlabel_{i}::{orig}"))
    _target_keys.append(f"xlabel_{i}::{orig}")
    # Value-label text target
    _target_options.append((f"Bar {i+1} \u2014 value-label text", f"vlabel_text_{i}::{orig}"))
    _target_keys.append(f"vlabel_text_{i}::{orig}")

_target_idx = st.selectbox(
    "Insert symbols into:",
    range(len(_target_options)),
    format_func=lambda x: _target_options[x][0],
    index=0,
)
_target_key = _target_keys[_target_idx]
_target_label, _target_field = _target_options[_target_idx]

st.markdown(f"**Current target:** {_target_label}")
st.markdown(
    f"&nbsp;&nbsp;Current value: "
    f"`{st.session_state.get(_target_key, '')}`"
)

# Render the symbol groups as rows of buttons (8 per row)
for grp_name, symbols in SYMBOL_GROUPS.items():
    st.markdown(f"**{grp_name}**")
    for row_start in range(0, len(symbols), 8):
        row = symbols[row_start:row_start + 8]
        btn_cols = st.columns(8)
        for j, (display, snippet) in enumerate(row):
            global_j = row_start + j
            if btn_cols[j].button(
                display,
                key=f"sym_{_target_key}_{grp_name}_{global_j}",
                use_container_width=True,
            ):
                st.session_state[_target_key] = st.session_state.get(_target_key, "") + snippet
                st.rerun()



# Render one expander per bar
for i, orig in enumerate(_default_categories):
    val = _default_values[i]
    auto_vtxt = val_label_fmt.format(val) if not pd.isna(val) else ""
    with st.expander(
        f"Bar {i+1}: {orig}",
        expanded=(i < 2),
    ):
        cc1, cc2 = st.columns(2)
        with cc1:
            new_xlab = st.text_input(
                "X-tick label",
                value=st.session_state[f"xlabel_{i}::{orig}"],
                key=f"xlabel_{i}::{orig}",
                help=f"Original: {orig}",
            )
            edited_labels.append(new_xlab)
        with cc2:
            new_vshow = st.checkbox(
                "Show value label on this bar",
                value=st.session_state[f"vlabel_show_{i}::{orig}"],
                key=f"vlabel_show_{i}::{orig}",
            )
            val_label_show.append(new_vshow)

        new_vtxt = st.text_input(
            f"Custom value label text (empty = auto: '{auto_vtxt}')",
            value=st.session_state[f"vlabel_text_{i}::{orig}"],
            key=f"vlabel_text_{i}::{orig}",
            help=(
                f"Leave empty to use the global format '{val_label_fmt}' applied to "
                f"the value ({auto_vtxt}). LaTeX math like $\\\\pm$ 0.2 works."
            ),
        )
        val_label_texts.append(new_vtxt)

        cc3, cc4 = st.columns(2)
        with cc3:
            color_ov = st.checkbox(
                "Override bar color (bypass W_k colormap)",
                value=st.session_state[f"bar_color_override_{i}::{orig}"],
                key=f"bar_color_override_{i}::{orig}",
            )
            if color_ov:
                color_val = st.color_picker(
                    "Bar color",
                    value=st.session_state[f"bar_color_val_{i}::{orig}"],
                    key=f"bar_color_val_{i}::{orig}",
                )
                bar_color_overrides.append(color_val)
            else:
                bar_color_overrides.append(None)
        with cc4:
            alpha_ov = st.checkbox(
                "Override bar alpha",
                value=st.session_state[f"bar_alpha_override_{i}::{orig}"],
                key=f"bar_alpha_override_{i}::{orig}",
            )
            if alpha_ov:
                alpha_val = st.slider(
                    "Bar alpha", 0.1, 1.0,
                    st.session_state[f"bar_alpha_val_{i}::{orig}"], 0.05,
                    key=f"bar_alpha_val_{i}::{orig}",
                )
                bar_alpha_overrides.append(alpha_val)
            else:
                bar_alpha_overrides.append(None)

if st.button("Reset all per-bar customizations"):
    for i, orig in enumerate(_default_categories):
        for prefix in (
            "xlabel", "vlabel_text", "vlabel_show",
            "bar_color_override", "bar_color_val",
            "bar_alpha_override", "bar_alpha_val",
        ):
            k = f"{prefix}_{i}::{orig}"
            if k in st.session_state:
                del st.session_state[k]
    st.rerun()

# ----------------------------------------------------------------------------- #
#  Build the figure
# ----------------------------------------------------------------------------- #
def build_figure() -> plt.Figure:
    mpl.rcParams["font.family"] = font_family

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.set_facecolor(facecol)
    ax.set_facecolor(ax_facecol)

    # Use the user-edited X-tick labels if available, else fall back to the column.
    try:
        edited = list(edited_labels)
    except NameError:
        edited = None
    raw_categories = df[cat_col].astype(str).tolist()
    if edited is not None and len(edited) == len(raw_categories):
        categories = list(edited)
    else:
        categories = list(raw_categories)

    values     = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    weights    = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)

    # Mask NaNs so matplotlib doesn't choke
    valid = ~np.isnan(values)
    categories = [c for c, v in zip(categories, valid) if v]
    values  = values[valid]
    weights = weights[valid]

    wmin = float(np.nanmin(weights)) if cb_vmin is None else float(cb_vmin)
    wmax = float(np.nanmax(weights)) if cb_vmax is None else float(cb_vmax)
    if wmax - wmin == 0:
        wmax = wmin + 1.0

    cmap = mpl.colormaps.get_cmap(cmap_name)
    if cmap_reverse:
        cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=wmin, vmax=wmax)
    colors = [cmap(norm(w)) for w in weights]

    # Apply per-bar color overrides (bypass W_k colormap)
    try:
        _overrides_c = list(bar_color_overrides)
    except NameError:
        _overrides_c = []
    for i, ov in enumerate(_overrides_c):
        if ov is not None and i < len(colors):
            colors[i] = ov

    # Per-bar alpha overrides
    try:
        _overrides_a = list(bar_alpha_overrides)
    except NameError:
        _overrides_a = []
    bar_alphas = [bar_alpha] * len(values)
    for i, ov in enumerate(_overrides_a):
        if ov is not None and i < len(bar_alphas):
            bar_alphas[i] = ov

    hatch = bar_hatch if bar_hatch else None
    bars = ax.bar(
        categories, values,
        width=bar_width,
        color=colors,
        edgecolor=edge_color,
        linewidth=edge_width,
        alpha=bar_alpha,  # global alpha (matplotlib applies this to all bars)
        hatch=hatch,
    )
    # Apply per-bar alpha after the fact (matplotlib's bar() doesn't accept a list for alpha)
    for b, a in zip(bars, bar_alphas):
        b.set_alpha(a)

    # Value labels on top of bars (per-bar control)
    # Use the global show_val_labels toggle, but also respect per-bar show/hide
    # and per-bar custom text overrides.
    try:
        _vtxts  = list(val_label_texts)
        _vshows = list(val_label_show)
    except NameError:
        _vtxts  = []
        _vshows = []
    ymax = float(np.nanmax(values)) if len(values) else 1.0
    for i, (b, v) in enumerate(zip(bars, values)):
        if np.isnan(v):
            continue
        this_show = show_val_labels
        if i < len(_vshows):
            this_show = this_show and bool(_vshows[i])
        if not this_show:
            continue
        if i < len(_vtxts) and _vtxts[i].strip() != "":
            txt = _vtxts[i]
        else:
            txt = val_label_fmt.format(v)
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + ymax * val_label_off,
            txt,
            ha="center", va="bottom",
            fontsize=tick_size, family=font_family,
        )

    # Labels & title
    ax.set_title(title_text, fontsize=title_size, family=font_family,
                 weight=title_weight, pad=14)
    ax.set_xlabel(xlabel_txt, fontsize=label_size, family=font_family)
    ax.set_ylabel(ylabel_txt, fontsize=label_size, family=font_family)

    ax.tick_params(
        axis="both",
        labelsize=tick_size,
        direction=tick_dir,
        length=tick_length,
        width=tick_width,
        top=tick_top, bottom=tick_bottom, left=tick_left, right=tick_right,
    )
    # Enhanced X-tick label styling (size / weight / color / alignment / math)
    _xt_size = tick_size if x_tick_size is None else x_tick_size
    for label, txt in zip(ax.get_xticklabels(), categories):
        label.set_rotation(xtick_rot)
        label.set_family(font_family)
        label.set_fontsize(_xt_size)
        label.set_fontweight(x_tick_weight)
        label.set_color(x_tick_color)
        label.set_horizontalalignment(x_tick_ha)
        if x_tick_use_math:
            label.set_text(f"${txt}$")
        else:
            label.set_text(txt)
    for label in ax.get_yticklabels():
        label.set_rotation(ytick_rot)
        label.set_family(font_family)

    # Spines: visibility + width + position
    for side, vis in spine_visible.items():
        ax.spines[side].set_visible(vis)
        ax.spines[side].set_linewidth(axes_linewidth)
        if not vis:
            continue
        if spine_pos_mode == "zero":
            ax.spines[side].set_position("zero")
        elif spine_pos_mode == "axes (0-1)":
            ax.spines[side].set_position(("axes", 0.0 if side in ("left", "bottom") else 1.0))
        elif spine_pos_mode == "data":
            ax.spines[side].set_position(("data", 0.0))
        elif spine_pos_mode == "outward (default)":
            ax.spines[side].set_position(("outward", spine_offset))

    # Y limits
    if not y_auto:
        ax.set_ylim(y_min, y_max)

    # Grid
    if show_grid:
        ax.grid(True, axis=grid_axis, alpha=grid_alpha, linewidth=axes_linewidth * 0.3)
    else:
        ax.grid(False)

    # Colorbar
    if show_cb:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(
            sm, ax=ax,
            orientation=cb_orientation,
            shrink=cb_shrink, pad=cb_pad, aspect=cb_aspect,
        )
        cb.set_label(cb_label, fontsize=cb_size, family=font_family)
        cb.ax.tick_params(labelsize=tick_size)
        cb.outline.set_linewidth(cb_outline_w)

    fig.tight_layout()
    return fig


fig = build_figure()
st.pyplot(fig, use_container_width=False)

# ----------------------------------------------------------------------------- #
#  Download buttons
# ----------------------------------------------------------------------------- #
st.subheader("Download")
col1, col2, col3 = st.columns(3)


def _buf(fmt: str) -> io.BytesIO:
    b = io.BytesIO()
    fig.savefig(b, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    b.seek(0)
    return b


with col1:
    st.download_button("PNG", _buf("png"), "bar_chart.png", "image/png")
with col2:
    st.download_button("PDF", _buf("pdf"), "bar_chart.pdf", "application/pdf")
with col3:
    st.download_button("SVG", _buf("svg"), "bar_chart.svg", "image/svg+xml")

# ----------------------------------------------------------------------------- #
#  Show the matplotlib source snippet for the current settings (optional)
# ----------------------------------------------------------------------------- #
with st.expander("Show equivalent matplotlib code (read-only)"):
    code = f"""import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams['font.family'] = {font_family!r}

df = pd.read_csv('q3cs-sigmas-qdwa-weights.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=({fig_w}, {fig_h}), dpi={dpi})

cmap = mpl.colormaps.get_cmap({cmap_name!r}){'_rev = cmap.reversed()' if cmap_reverse else ''}
norm = mpl.colors.Normalize(vmin={cb_vmin}, vmax={cb_vmax})
colors = [cmap(norm(w)) for w in df[{weight_col!r}]]

bars = ax.bar(df[{cat_col!r}], df[{value_col!r}],
              width={bar_width}, color=colors,
              edgecolor={edge_color!r}, linewidth={edge_width}, alpha={bar_alpha})

# Per-bar overrides (color, alpha, value-label text/visibility)
_bar_color_ov = {bar_color_overrides!r}
_bar_alpha_ov = {bar_alpha_overrides!r}
_val_texts    = {val_label_texts!r}
_val_show     = {val_label_show!r}
for i, b in enumerate(bars):
    if i < len(_bar_color_ov) and _bar_color_ov[i] is not None:
        b.set_color(_bar_color_ov[i])
    if i < len(_bar_alpha_ov) and _bar_alpha_ov[i] is not None:
        b.set_alpha(_bar_alpha_ov[i])
_xlabels = {edited_labels!r}
ax.set_xticks(range(len(_xlabels)))
ax.set_xticklabels(_xlabels)
_ymax = df[{value_col!r}].max()
for i, (b, v) in enumerate(zip(bars, df[{value_col!r}])):
    if not _val_show[i]:
        continue
    if _val_texts[i].strip():
        _t = _val_texts[i]
    else:
        _t = "{val_label_fmt}".format(v)
    ax.text(b.get_x() + b.get_width()/2, v + _ymax*{val_label_off},
            _t, ha="center", va="bottom", fontsize={tick_size})

ax.set_title({title_text!r}, fontsize={title_size}, weight={title_weight!r})
ax.set_xlabel({xlabel_txt!r}, fontsize={label_size})
ax.set_ylabel({ylabel_txt!r}, fontsize={label_size})
ax.tick_params(labelsize={tick_size}, direction={tick_dir!r},
               length={tick_length}, width={tick_width},
               top={tick_top}, bottom={tick_bottom},
               left={tick_left}, right={tick_right})
# Enhanced X-tick labels
_xt_size = {tick_size if x_tick_size is None else x_tick_size!r}
_xt_labels = {edited_labels!r}
for _lbl, _txt in zip(ax.get_xticklabels(), _xt_labels):
    _lbl.set_fontsize(_xt_size)
    _lbl.set_fontweight({x_tick_weight!r})
    _lbl.set_color({x_tick_color!r})
    _lbl.set_horizontalalignment({x_tick_ha!r})
    if {x_tick_use_math}:
        _lbl.set_text(f"${{_txt}}$")
for side, vis in {spine_visible!r}.items():
    ax.spines[side].set_visible(vis)
    ax.spines[side].set_linewidth({axes_linewidth})

cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                  orientation={cb_orientation!r}, shrink={cb_shrink}, pad={cb_pad})
cb.set_label({cb_label!r}, fontsize={cb_size})
fig.tight_layout()
plt.show()
"""
    st.code(code, language="python")
