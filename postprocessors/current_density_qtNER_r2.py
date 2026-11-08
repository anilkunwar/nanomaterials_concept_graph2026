"""
Streamlit app: Histogram studio for quantitative NER outputs
================================================================
Plots a histogram (bar chart of binned value counts) for the quantitative
named-entity-recognition (qtNER) outputs of electrodeposition / electroplating
experiments, with focus on **current density (mA/cm²)** for nanotwinned Cu
preparation, matching the reference figure `j.png`.

The data source is `qtNER-1.csv` — the raw extracted metrics from papers
(columns: doc_id, metric_type, raw_value, raw_unit, normalized_value, etc.).
You can filter by metric_type (default: current_density) and by raw_unit,
optionally convert units (A/dm² → mA/cm² with factor 10), and tweak every
visual property in the sidebar.

Run it with:
    streamlit run histogram_app.py
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
    page_title="qtNER Histogram Studio — Current Density",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 qtNER Histogram Studio — Current Density")
st.markdown(
    "Histogram of quantitative NER outputs for electrodeposition / electroplating "
    "experiments (nanotwinned Cu preparation). Default view: **current density "
    "(mA/cm²)** distribution, matching the reference figure `j.png`."
)

# ----------------------------------------------------------------------------- #
#  Helpers
# ----------------------------------------------------------------------------- #
DEFAULT_CSV  = Path(__file__).with_name("qtNER-1.csv")
SUMMARY_CSV  = Path(__file__).with_name("qtNER-2.csv")

# Bundled fallback: the 17 current_density rows from qtNER-1.csv, embedded directly
# in the code so the default view works even if the bundled CSV file is missing
# (e.g. when only histogram_app.py is downloaded without the CSV).
# For the other metric types (annealing_temperature, crystallite_size), upload the
# full qtNER-1.csv via the sidebar.
FALLBACK_CSV = """
doc_id,metric_type,metric_display,raw_value,raw_unit,normalized_value,normalized_unit,material,material_role,context,confidence,method
758,current_density,Current Density,2.4,mA/cm2,2.4,mA/cm2,,,"ibited the highest offload current (mA) of 15 mA, maximum power of 14.19 mW and ...",0.85,regex
992,current_density,Current Density,18.65,mA/cm2,18.65,mA/cm2,,,"an ETL exhibited a power conversion efficiency (PEC) of 11.46%, a short-circuit ...",0.85,regex
1258,current_density,Current Density,10.0,mA/cm2,10.0,mA/cm2,,,"0.5 M H2SO4 solution, the catalyst achieved an overpotential of only 32 mV at a ...",0.85,regex
1273,current_density,Current Density,30.0,mA/cm2,30.0,mA/cm2,,,lectrolyte solution with varying concentrations of silver nitrate at a constant ...,0.85,regex
1373,current_density,Current Density,10.0,mA/cm2,10.0,mA/cm2,,," is smaller than pure Ru, Nb1Ru1 sample achieves an overpotential of 39 mV at a ...",0.85,regex
1378,current_density,Current Density,10.0,mA/cm2,10.0,mA/cm2,,,s by varying the current density from 1 mA/cm2 to 50 mA/cm2. The coating with a ...,0.85,regex
1486,current_density,Current Density,24.3,mA/cm2,24.3,mA/cm2,,,"ciency (PCE) of 10.4%, with an open-circuit voltage of 0.701 V, a short-circuit ...",0.85,regex
1614,current_density,Current Density,300.0,mA/cm2,300.0,mA/cm2,,,graphene was obtained at a graphene concentration (C Gr) of 0.1 g/L and a pulse ...,0.85,regex
1669,current_density,Current Density,55.0,mA/cm2,55.0,mA/cm2,,,o between (Ni/Cu/N) MOF and G-graphite (2:1) is the best with an oxidation peak ...,0.85,regex
1708,current_density,Current Density,15.6,A/dm2,15.6,A/dm2,,,r-rent per 1 dm2 of the anode plane. The electrolysis process is continued at a ...,0.85,regex
1733,current_density,Current Density,10.0,mA/cm2,10.0,mA/cm2,,,ytic efficiency for HER. The overpotential of HER required to reach a catalytic ...,0.85,regex
1823,current_density,Current Density,10.0,mA/cm2,10.0,mA/cm2,,,"n investigated, which results in a further lowered overpotential of 265 mV at a ...",0.85,regex
2042,current_density,Current Density,4.24,mA/cm2,4.24,mA/cm2,,,l studies on CuO-NS reveal intriguing methanol electrooxidation properties with ...,0.85,regex
2262,current_density,Current Density,400.0,A/dm2,400.0,A/dm2,,,"icrohardness increases by 81%, and the tensile strength increases by 51%, using ...",0.85,regex
2328,current_density,Current Density,400.0,A/dm2,400.0,A/dm2,,, fine microstructure comprising nanosized grains with a size of 28 nm at a peak ...,0.85,regex
2579,current_density,Current Density,546.81,mA/cm2,546.81,mA/cm2,,,gCu-20 % electrocatalyst shows a peak power density of 230 mW/cm2 and a maximum ...,0.85,regex
2751,current_density,Current Density,5.3,mA/cm2,5.3,mA/cm2,,,"gh-order products i.e., CH4, C2H4, C2H5OH and CH3COOH reaches 55.3% with a high ...",0.85,regex
"""


@st.cache_data
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def load_fallback() -> pd.DataFrame:
    """Parse the embedded FALLBACK_CSV string into a DataFrame."""
    return pd.read_csv(io.StringIO(FALLBACK_CSV), encoding="utf-8-sig")


def list_font_families() -> list[str]:
    return sorted({f.name for f in font_manager.fontManager.ttflist})


def all_colormaps() -> list[str]:
    return sorted(mpl.colormaps())


# ----------------------------------------------------------------------------- #
#  1. Data input & filtering  ----------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("1️⃣ Data input & filter")
src = st.sidebar.radio(
    "Data source",
    ["Default CSV (bundled qtNER-1.csv)", "Upload CSV", "Manual entry (one value per line)"],
    index=0,
    help="qtNER-1.csv holds the raw extracted metrics from papers. Upload your own CSV "
         "with at least a `metric_type`, `raw_value`, and `raw_unit` column, or type "
         "values by hand.",
)

if src == "Default CSV (bundled qtNER-1.csv)":
    if DEFAULT_CSV.exists():
        df = load_csv(DEFAULT_CSV)
    else:
        # Bundled CSV not found next to this script — fall back to the embedded
        # 17 current_density rows so the default view still works.
        df = load_fallback()
        st.sidebar.warning(
            f"⚠️ Bundled `qtNER-1.csv` not found next to `{Path(__file__).name}` "
            f"(looked in `{DEFAULT_CSV.parent}`). Using the embedded fallback "
            f"(17 current_density rows) so the default view works. To also see "
            f"`annealing_temperature` and `crystallite_size`, upload the full "
            f"`qtNER-1.csv` via the **Upload CSV** option above."
        )
elif src == "Upload CSV":
    up = st.sidebar.file_uploader("Choose a CSV file", type=["csv", "tsv"])
    if up is not None:
        try:
            df = pd.read_csv(up, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(up)
    else:
        df = load_csv(DEFAULT_CSV) if DEFAULT_CSV.exists() else pd.DataFrame()
        st.sidebar.info("No file uploaded yet — using the bundled qtNER-1.csv.")
else:
    st.sidebar.markdown("Type values, one per line. Optionally append `| unit` (default mA/cm2).")
    raw_txt = st.sidebar.text_area(
        "Manual data",
        "2.4\n18.65\n10\n30\n10\n10\n24.3\n300\n55\n15.6\n10\n10\n4.24\n400\n400\n546.81\n5.3",
        height=160,
    )
    rows = []
    for line in raw_txt.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        try:
            v = float(parts[0])
            u = parts[1] if len(parts) > 1 else "mA/cm2"
            rows.append({"metric_type": "current_density", "raw_value": v, "raw_unit": u,
                         "normalized_value": v, "normalized_unit": u,
                         "metric_display": "Current Density", "doc_id": None,
                         "material": "", "material_role": "", "context": "",
                         "confidence": None, "method": ""})
        except ValueError:
            pass
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["metric_type", "raw_value", "raw_unit", "normalized_value", "normalized_unit"])

st.subheader("Data preview")
st.dataframe(df, use_container_width=True, hide_index=True)

# --- Summary statistics panel (from qtNER-2.csv if available) ----------------
if SUMMARY_CSV.exists():
    summ = load_csv(SUMMARY_CSV)
    st.sidebar.markdown("**Summary statistics (qtNER-2.csv)**")
    st.sidebar.dataframe(summ, use_container_width=True, hide_index=True)

# --- Metric type filter ------------------------------------------------------
metric_types = sorted(df["metric_type"].dropna().unique().tolist()) if "metric_type" in df.columns else []
if not metric_types:
    st.error("No `metric_type` column found in the data.")
    st.stop()
default_mt = "current_density" if "current_density" in metric_types else metric_types[0]
metric_type = st.sidebar.selectbox("Metric type", metric_types,
                                   index=metric_types.index(default_mt))

# --- Value column choice (raw vs normalized) --------------------------------
val_col_options = [c for c in ["raw_value", "normalized_value"] if c in df.columns]
default_vc = "raw_value" if "raw_value" in val_col_options else val_col_options[0]
value_col = st.sidebar.selectbox("Value column", val_col_options,
                                 index=val_col_options.index(default_vc),
                                 help="Use `raw_value` to match the reference j.png (no unit "
                                      "conversion). Use `normalized_value` if your data is "
                                      "already in a common unit.")

# --- Unit filter ------------------------------------------------------------
unit_col = "raw_unit" if value_col == "raw_value" else "normalized_unit"
units = ["(all)"] + sorted(df[unit_col].dropna().unique().tolist()) if unit_col in df.columns else ["(all)"]
unit_filter = st.sidebar.selectbox(f"Filter by {unit_col}", units, index=0)

# --- Optional unit conversion ----------------------------------------------
convert_unit = st.sidebar.checkbox("Apply unit conversion", value=False,
    help="Convert all values to a single target unit before plotting. "
         "Useful when the data mixes mA/cm² and A/dm².")
if convert_unit:
    target_unit = st.sidebar.text_input("Target unit", value="mA/cm2")
    conv_factor = st.sidebar.number_input(
        "Multiplicative factor (target = source × factor)",
        value=10.0, format="%.4f",
        help="Example: A/dm² → mA/cm² uses factor 10 (1 A/dm² = 1000 mA / 100 cm² = 10 mA/cm²)."
    )
    source_unit_for_conv = st.sidebar.text_input("Source unit to convert FROM",
                                                 value="A/dm2",
                                                 help="Only rows whose unit matches this string "
                                                      "will be multiplied by the factor.")
else:
    target_unit = None
    conv_factor = 1.0
    source_unit_for_conv = None

# Apply the filter to get the values to plot
sub = df[df["metric_type"] == metric_type].copy()
if unit_filter != "(all)" and unit_col in sub.columns:
    sub = sub[sub[unit_col] == unit_filter]
if convert_unit and unit_col in sub.columns:
    sub[value_col] = sub.apply(
        lambda r: float(r[value_col]) * conv_factor
                  if str(r[unit_col]) == source_unit_for_conv else float(r[value_col]),
        axis=1,
    )

values = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy(dtype=float)

st.subheader(f"Filtered data — {metric_type} ({len(values)} values)")
if len(values):
    st.write(f"min={values.min():.4g}, max={values.max():.4g}, "
             f"mean={values.mean():.4g}, std={values.std():.4g}")
    st.dataframe(sub[[c for c in ["doc_id", value_col, unit_col, "context"]
                      if c in sub.columns]].reset_index(drop=True),
                 use_container_width=True, hide_index=True)
else:
    st.warning("No values match the current filters.")

# ----------------------------------------------------------------------------- #
#  2. Labels & title  -----------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("2️⃣ Labels & title")
_title_default = f"{metric_type.replace('_', ' ').title()} Distribution"
# Choose the most common unit in the (filtered) subset as the default x-axis label,
# so the histogram reads "Value (mA/cm2)" for current_density by default.
_default_unit_for_label = ""
if unit_col in sub.columns and len(sub):
    vc = sub[unit_col].value_counts()
    if len(vc):
        _default_unit_for_label = str(vc.idxmax())
_x_default = f"Value ({target_unit or _default_unit_for_label})"
title_text  = st.sidebar.text_input("Chart title", _title_default)
xlabel_txt = st.sidebar.text_input("X-axis label", _x_default)
ylabel_txt = st.sidebar.text_input("Y-axis label", "count")
show_val_labels = st.sidebar.checkbox("Show count labels on bars", value=False)
val_label_fmt   = st.sidebar.text_input("Count label format", "{:.0f}")
val_label_off   = st.sidebar.slider("Count label vertical offset", 0.0, 2.0, 0.2, 0.05)

# ----------------------------------------------------------------------------- #
#  3. Fonts  --------------------------------------------------------------------
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
title_weight  = st.sidebar.selectbox("Title weight",
                                    ["normal", "bold", "light", "semibold"], index=1)

# ----------------------------------------------------------------------------- #
#  4. Histogram bins  -----------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("4️⃣ Histogram bins")
bin_mode = st.sidebar.radio(
    "Bin mode",
    ["Number of bins", "Bin width", "Custom bin edges"],
    index=0, horizontal=True,
)
if bin_mode == "Number of bins":
    n_bins = st.sidebar.slider("Number of bins", 1, 200, 11)
    bin_width = None
    custom_bins_str = ""
elif bin_mode == "Bin width":
    bin_width = st.sidebar.number_input(
        "Bin width", value=50.0, format="%.4g",
        help="E.g. 50 matches the reference figure `j.png` (bins of 50 mA/cm²)."
    )
    n_bins = None
    custom_bins_str = ""
else:
    custom_bins_str = st.sidebar.text_input(
        "Custom bin edges (comma-separated)",
        value="0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600",
        help="Edges define bin boundaries. n bins = len(edges) - 1."
    )
    n_bins = None
    bin_width = None

bin_range_auto = st.sidebar.checkbox("Auto bin range", value=True)
if not bin_range_auto:
    bin_lo = st.sidebar.number_input("Bin range low", value=0.0)
    bin_hi = st.sidebar.number_input("Bin range high", value=600.0)
else:
    bin_lo = bin_hi = None

density = st.sidebar.checkbox("Density (PDF instead of count)", value=False)

# ----------------------------------------------------------------------------- #
#  5. Bar color  ----------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("5️⃣ Bar color")
color_mode = st.sidebar.radio(
    "Color mode",
    ["Solid color", "Colormap by bin index", "Colormap by bin center"],
    index=0, horizontal=True,
)
cmaps = all_colormaps()
if color_mode != "Solid color":
    cmap_name = st.sidebar.selectbox("Colormap", cmaps, index=cmaps.index("viridis"))
    cmap_reverse = st.sidebar.checkbox("Reverse colormap", value=False)
else:
    cmap_name = None
    cmap_reverse = False
solid_color = st.sidebar.color_picker("Solid color", "#1f77b4")
edge_color  = st.sidebar.color_picker("Bar edge color", "#000000")
edge_width  = st.sidebar.slider("Bar edge thickness", 0.0, 5.0, 0.8, 0.1)
bar_alpha   = st.sidebar.slider("Bar alpha", 0.1, 1.0, 1.0, 0.05)
bar_hatch   = st.sidebar.selectbox("Bar hatch",
                                   ["", "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"],
                                   index=0)

# ----------------------------------------------------------------------------- #
#  6. Axes & ticks  -------------------------------------------------------------
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

# X-tick label font enhancement
st.sidebar.markdown("**X-tick label font enhancement**")
x_tick_size_override = st.sidebar.checkbox("Custom X-tick label size", value=False,
    help="Independent of the global tick-label size above.")
x_tick_size = st.sidebar.slider("X-tick label size", 4, 48, 14) if x_tick_size_override else None
x_tick_weight = st.sidebar.selectbox("X-tick label weight",
                                    ["normal", "bold", "light", "semibold"], index=1)
x_tick_color  = st.sidebar.color_picker("X-tick label color", "#000000")
x_tick_ha     = st.sidebar.selectbox("X-tick label horizontal alignment",
                                    ["center", "right", "left"], index=0)

# ----------------------------------------------------------------------------- #
#  7. Spines  -------------------------------------------------------------------
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
spine_offset = st.sidebar.slider("Outward offset (pts)", -50.0, 50.0, 0.0, 1.0) \
               if spine_pos_mode == "outward (default)" else None

# ----------------------------------------------------------------------------- #
#  8. Figure  -------------------------------------------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("8️⃣ Figure")
fig_w   = st.sidebar.slider("Figure width (in)", 3.0, 16.0, 7.0, 0.25)
fig_h   = st.sidebar.slider("Figure height (in)", 2.0, 14.0, 5.0, 0.25)
dpi     = st.sidebar.slider("DPI", 72, 600, 150, 10)
facecol = st.sidebar.color_picker("Figure face color", "#FFFFFF")
ax_facecol = st.sidebar.color_picker("Axes face color", "#FFFFFF")
y_auto  = st.sidebar.checkbox("Auto Y-limits", value=True)
if not y_auto:
    y_min = st.sidebar.number_input("Y min", value=0.0)
    y_max = st.sidebar.number_input("Y max", value=15.0)
else:
    y_min = y_max = None
x_auto  = st.sidebar.checkbox("Auto X-limits", value=True)
if not x_auto:
    x_min = st.sidebar.number_input("X min", value=0.0)
    x_max = st.sidebar.number_input("X max", value=600.0)
else:
    x_min = x_max = None
show_grid = st.sidebar.checkbox("Show grid", value=True,
    help="Reference figure j.png uses a white-on-grid background.")
grid_axis = st.sidebar.selectbox("Grid axis", ["both", "x", "y"], index=0) if show_grid else "both"
grid_alpha = st.sidebar.slider("Grid alpha", 0.05, 1.0, 0.4, 0.05) if show_grid else 0.4
grid_linestyle = st.sidebar.selectbox("Grid line style",
                                      ["-", "--", ":", "-."], index=0) if show_grid else "-"

# ----------------------------------------------------------------------------- #
#  9. Colorbar (only shown in colormap modes)  ----------------------------------
# ----------------------------------------------------------------------------- #
st.sidebar.title("9️⃣ Colorbar")
show_cb = st.sidebar.checkbox("Show colorbar", value=(color_mode != "Solid color"))
cb_orientation = st.sidebar.selectbox("Colorbar orientation", ["vertical", "horizontal"])
cb_shrink      = st.sidebar.slider("Colorbar shrink", 0.2, 2.0, 0.8, 0.05)
cb_pad         = st.sidebar.slider("Colorbar pad", 0.0, 1.0, 0.04, 0.01)
cb_aspect      = st.sidebar.slider("Colorbar aspect", 5, 60, 20, 1)
cb_outline_w   = st.sidebar.slider("Colorbar outline width", 0.0, 5.0, 0.5, 0.1)
cb_label       = st.sidebar.text_input("Colorbar label",
                                       "bin index" if color_mode == "Colormap by bin index"
                                       else "bin center value")

# ----------------------------------------------------------------------------- #
#  Build the figure
# ----------------------------------------------------------------------------- #
def build_figure() -> plt.Figure:
    mpl.rcParams["font.family"] = font_family

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.set_facecolor(facecol)
    ax.set_facecolor(ax_facecol)

    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=label_size, family=font_family)
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    # Determine bin edges
    if bin_mode == "Number of bins":
        if bin_range_auto:
            bins = n_bins
        else:
            bins = np.linspace(bin_lo, bin_hi, n_bins + 1)
    elif bin_mode == "Bin width":
        lo = values.min() if bin_range_auto else bin_lo
        hi = values.max() if bin_range_auto else bin_hi
        bins = np.arange(lo, hi + bin_width, bin_width)
    else:
        try:
            bins = np.array([float(x.strip()) for x in custom_bins_str.split(",") if x.strip()])
            bins = np.sort(bins)
        except ValueError:
            bins = 10
        if bin_range_auto or len(bins) < 2:
            bins = bins if len(bins) >= 2 else 10

    # Compute histogram counts (we draw with ax.bar so we can color per-bin)
    counts, edges = np.histogram(values, bins=bins, density=density)
    centers = (edges[:-1] + edges[1:]) / 2
    widths  = np.diff(edges)

    # Per-bar colors
    if color_mode == "Solid color":
        colors = [solid_color] * len(counts)
        norm = None
        cmap = None
    else:
        cmap = mpl.colormaps.get_cmap(cmap_name)
        if cmap_reverse:
            cmap = cmap.reversed()
        if color_mode == "Colormap by bin index":
            if len(counts) > 1:
                norm = mpl.colors.Normalize(vmin=0, vmax=len(counts) - 1)
            else:
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
            colors = [cmap(norm(i)) for i in range(len(counts))]
        else:  # by bin center
            norm = mpl.colors.Normalize(vmin=edges.min(), vmax=edges.max())
            colors = [cmap(norm(c)) for c in centers]

    hatch = bar_hatch if bar_hatch else None
    bars = ax.bar(
        centers, counts, width=widths,
        color=colors, edgecolor=edge_color, linewidth=edge_width,
        alpha=bar_alpha, hatch=hatch, align="center",
    )

    # Count labels on top of bars
    if show_val_labels:
        ymax = float(np.nanmax(counts)) if len(counts) else 1.0
        for b, c in zip(bars, counts):
            if c == 0:
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                c + ymax * val_label_off,
                val_label_fmt.format(c),
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
    _xt_size = tick_size if x_tick_size is None else x_tick_size
    for label in ax.get_xticklabels():
        label.set_rotation(xtick_rot)
        label.set_family(font_family)
        label.set_fontsize(_xt_size)
        label.set_fontweight(x_tick_weight)
        label.set_color(x_tick_color)
        label.set_horizontalalignment(x_tick_ha)
    for label in ax.get_yticklabels():
        label.set_rotation(ytick_rot)
        label.set_family(font_family)

    # Spines
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

    # Limits
    if not y_auto:
        ax.set_ylim(y_min, y_max)
    if not x_auto:
        ax.set_xlim(x_min, x_max)

    # Grid
    if show_grid:
        ax.grid(True, axis=grid_axis, alpha=grid_alpha,
                linewidth=axes_linewidth * 0.3, linestyle=grid_linestyle)
    else:
        ax.grid(False)

    # Colorbar
    if show_cb and color_mode != "Solid color" and norm is not None:
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
    st.download_button("PNG", _buf("png"), "histogram.png", "image/png")
with col2:
    st.download_button("PDF", _buf("pdf"), "histogram.pdf", "application/pdf")
with col3:
    st.download_button("SVG", _buf("svg"), "histogram.svg", "image/svg+xml")

# ----------------------------------------------------------------------------- #
#  Equivalent matplotlib code (read-only)
# ----------------------------------------------------------------------------- #
with st.expander("Show equivalent matplotlib code (read-only)"):
    code = f"""import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams['font.family'] = {font_family!r}

df = pd.read_csv('qtNER-1.csv', encoding='utf-8-sig')
sub = df[df['metric_type'] == {metric_type!r}]
values = pd.to_numeric(sub[{value_col!r}]).dropna().to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=({fig_w}, {fig_h}), dpi={dpi})
counts, edges = np.histogram(values, bins={n_bins if n_bins is not None else (repr(list(np.arange(0, 600.0, 50.0))) if bin_mode == 'Bin width' else custom_bins_str)},
                              density={density})
centers = (edges[:-1] + edges[1:]) / 2
widths  = np.diff(edges)
{f"cmap = mpl.colormaps.get_cmap({cmap_name!r}){'_rev = cmap.reversed()' if cmap_reverse else ''}" if color_mode != 'Solid color' else ''}
{f"norm = mpl.colors.Normalize(vmin=0, vmax=len(counts)-1)" if color_mode == 'Colormap by bin index' else ''}
{f"norm = mpl.colors.Normalize(vmin=edges.min(), vmax=edges.max())" if color_mode == 'Colormap by bin center' else ''}
{f"colors = [cmap(norm(i)) for i in range(len(counts))]" if color_mode == 'Colormap by bin index' else ''}
{f"colors = [cmap(norm(c)) for c in centers]" if color_mode == 'Colormap by bin center' else ''}
{f"colors = [{solid_color!r}] * len(counts)" if color_mode == 'Solid color' else ''}
ax.bar(centers, counts, width=widths, color=colors,
       edgecolor={edge_color!r}, linewidth={edge_width}, alpha={bar_alpha})

ax.set_title({title_text!r}, fontsize={title_size}, weight={title_weight!r})
ax.set_xlabel({xlabel_txt!r}, fontsize={label_size})
ax.set_ylabel({ylabel_txt!r}, fontsize={label_size})
ax.tick_params(labelsize={tick_size}, direction={tick_dir!r},
               length={tick_length}, width={tick_width},
               top={tick_top}, bottom={tick_bottom},
               left={tick_left}, right={tick_right})
{'ax.grid(True, axis=' + repr(grid_axis) + ', alpha=' + repr(grid_alpha) + ')' if show_grid else 'ax.grid(False)'}
fig.tight_layout()
plt.show()
"""
    st.code(code, language="python")
