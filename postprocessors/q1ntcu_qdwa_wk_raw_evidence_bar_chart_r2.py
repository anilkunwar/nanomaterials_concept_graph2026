"""
Streamlit app: W_k-weighted bar chart with full customization
================================================================
Recreates the published-style bar chart shown in `w_k_q2agnp_qdwa.jpg`:
    - bars colored by the W_k colormap (right-hand colorbar)
    - bar height = "Raw" evidence value
    - publication-grade typography, spines, ticks, etc.

Run it with:
    streamlit run bar_chart_app.py
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
    page_title="W_k Bar Chart Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 W_k-weighted Bar Chart Studio")
st.markdown(
    "Publication-grade bar chart whose bar colors are driven by the **W_k** "
    "weight column (right-hand colorbar), exactly like the reference figure. "
    "Every visual property can be tweaked in the sidebar."
)

# ----------------------------------------------------------------------------- #
#  Helpers
# ----------------------------------------------------------------------------- #
DEFAULT_CSV = Path(__file__).with_name("q1ntcu_qdwa_weights.csv")

FALLBACK_DF = pd.DataFrame(
    {
        "Rank": [1, 2, 3],
        "Category": ["Electrodeposited nt-Cu", "Defect-Engineered Ag", "Cu@Ag Core-Shell"],
        "W_k": [0.8495, 0.0753, 0.0753],
        "Raw": [6.33, 0.33, 0.33],
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
        "Electrodeited nt-Cu, 0.8495, 6.33\nDefect-Engineered Ag, 0.0753, 0.33\nCu@Ag Core-Shell, 0.0753, 0.33",
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
#  Editable x-axis tick labels  ------------------------------------------------
# ----------------------------------------------------------------------------- #
_default_categories = df[cat_col].astype(str).tolist()

st.subheader("✏️ Edit X-axis tick labels")
st.markdown(
    "Rename the label shown under each bar. The original category name is on the "
    "left; type any text on the right (LaTeX math like `$\\\\downarrow$-Ag` works)."
)

# Persist edits across reruns in session_state keyed by the original label,
# so changing an unrelated slider doesn't wipe the user's edits.
edited_labels: list[str] = []
_cols_per_row = 2
for i, orig in enumerate(_default_categories):
    key = f"xlabel_{i}::{orig}"
    if key not in st.session_state:
        st.session_state[key] = orig
    new_val = st.text_input(
        f"Bar {i+1}", value=st.session_state[key],
        key=key, help=f"Original: {orig}",
    )
    edited_labels.append(new_val)

# Also expose a compact "reset to original" button
if st.button("Reset X-axis labels to original"):
    for i, orig in enumerate(_default_categories):
        st.session_state[f"xlabel_{i}::{orig}"] = orig
    st.rerun()

# ----------------------------------------------------------------------------- #
#  2. Labels & title  -------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("2️⃣ Labels & title")
title_text  = st.sidebar.text_input("Chart title", "RAW EVIDENCE by CATEGORY")
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
    y_max = st.sidebar.number_input("Y max", value=7.0)
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

    hatch = bar_hatch if bar_hatch else None
    bars = ax.bar(
        categories, values,
        width=bar_width,
        color=colors,
        edgecolor=edge_color,
        linewidth=edge_width,
        alpha=bar_alpha,
        hatch=hatch,
    )

    # Value labels on top of bars
    if show_val_labels:
        ymax = float(np.nanmax(values)) if len(values) else 1.0
        for b, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + ymax * val_label_off,
                    val_label_fmt.format(v),
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

df = pd.read_csv('q1ntcu_qdwa_weights.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=({fig_w}, {fig_h}), dpi={dpi})

cmap = mpl.colormaps.get_cmap({cmap_name!r}){'_rev = cmap.reversed()' if cmap_reverse else ''}
norm = mpl.colors.Normalize(vmin={cb_vmin}, vmax={cb_vmax})
colors = [cmap(norm(w)) for w in df[{weight_col!r}]]

bars = ax.bar(df[{cat_col!r}], df[{value_col!r}],
              width={bar_width}, color=colors,
              edgecolor={edge_color!r}, linewidth={edge_width}, alpha={bar_alpha})

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
