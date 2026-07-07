#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nanomaterials-ConceptGraph v3.1: Enhanced Sunburst & Visualization Patch
==================================================================================
Surgical enhancements to the v3.0 codebase focusing on:
1. Sunburst symbol visibility (horizontal text, spaced symbols, custom hover)
2. Node centering improvements (hierarchical layout option)
3. Color contrast optimization for symbol readability
4. Hover template fixes for symbol-to-label mapping

Apply these as a drop-in replacement for the corresponding functions in v3.0.
"""

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import re
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from pyvis.network import Network

# ==========================================
# ENHANCED COLORMAP REGISTRY (v3.1)
# ==========================================
SUPPORTED_COLORMAPS = {
    "viridis": "Viridis", "plasma": "Plasma", "inferno": "Inferno", "magma": "Magma",
    "cividis": "Cividis", "turbo": "Turbo", "jet": "Jet", "rainbow": "Rainbow",
    "hsv": "Hsv", "nipy_spectral": "NipySpectral", "gist_rainbow": "GistRainbow",
    "coolwarm": "Coolwarm", "RdBu": "RdBu", "seismic": "Seismic", "Spectral": "Spectral",
    "tab10": "Set1", "tab20": "Set2", "tab20b": "Set3", "Accent": "Accent",
    "Dark2": "Dark2", "Paired": "Paired", "Pastel1": "Pastel1", "Pastel2": "Pastel2",
    "cubehelix": "Cubehelix", "bone": "Bone", "gray": "Gray", "pink": "Pink",
    "spring": "Spring", "summer": "Summer", "autumn": "Autumn", "winter": "Winter",
    "cool": "Cool", "hot": "Hot", "twilight": "Twilight", "copper": "Copper",
    "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "RdPu": "RdPu",
    "BuPu": "BuPu", "GnBu": "GnBu", "YlGnBu": "YlGnBu", "PuBuGn": "PuBuGn",
    "BuGn": "BuGn", "YlGn": "YlGn", "Greys": "Greys", "afmhot": "Afmhot",
    "gist_earth": "GistEarth", "terrain": "Terrain", "ocean": "Ocean"
}

def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    """Get n distinct hex colors from a matplotlib colormap."""
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
    except Exception:
        try:
            cmap = cm.get_cmap(cmap_name, n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
        except Exception:
            try:
                cmap = matplotlib.colormaps.get_cmap("viridis").resampled(n)
            except Exception:
                cmap = cm.get_cmap("viridis", n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


# ==========================================
# v3.1: ENHANCED THEME CONFIGURATION
# ==========================================
THEME_PRESETS = {
    "Bright (Default)": {
        "bg": "#ffffff", "font": "#1e293b", "tooltip_bg": "rgba(255,255,255,0.95)",
        "tooltip_border": "#cbd5e1", "tooltip_text": "#1e293b",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.45)",
        "edge_semantic": "rgba(251, 146, 60, 0.40)",
        "edge_bridge": "rgba(250, 204, 21, 0.55)",
        "edge_unknown": "rgba(148, 163, 184, 0.30)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b", "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.15)", "plotly_bg": "#ffffff", "plotly_paper": "#ffffff",
        "grid_color": "#e2e8f0", "axis_color": "#64748b"
    },
    "Dark": {
        "bg": "#0f172a", "font": "#e2e8f0", "tooltip_bg": "rgba(15, 23, 42, 0.95)",
        "tooltip_border": "#334155", "tooltip_text": "#e2e8f0",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.55)",
        "edge_semantic": "rgba(251, 146, 60, 0.50)",
        "edge_bridge": "rgba(250, 204, 21, 0.65)",
        "edge_unknown": "rgba(148, 163, 184, 0.40)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b", "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.6)", "plotly_bg": "#0f172a", "plotly_paper": "#0f172a",
        "grid_color": "#1e293b", "axis_color": "#94a3b8"
    },
    "Midnight": {
        "bg": "#020617", "font": "#f1f5f9", "tooltip_bg": "rgba(2, 6, 23, 0.97)",
        "tooltip_border": "#1e293b", "tooltip_text": "#f1f5f9",
        "edge_cooccurrence": "rgba(99, 102, 241, 0.55)",
        "edge_semantic": "rgba(236, 72, 153, 0.50)",
        "edge_bridge": "rgba(34, 211, 238, 0.65)",
        "edge_unknown": "rgba(71, 85, 105, 0.40)",
        "node_border": "#e2e8f0", "highlight_bg": "#f43f5e", "hover_bg": "#22d3ee",
        "shadow_color": "rgba(0,0,0,0.7)", "plotly_bg": "#020617", "plotly_paper": "#020617",
        "grid_color": "#0f172a", "axis_color": "#64748b"
    },
    "Warm": {
        "bg": "#fff7ed", "font": "#431407", "tooltip_bg": "rgba(255, 247, 237, 0.97)",
        "tooltip_border": "#fdba74", "tooltip_text": "#431407",
        "edge_cooccurrence": "rgba(234, 88, 12, 0.45)",
        "edge_semantic": "rgba(180, 83, 9, 0.40)",
        "edge_bridge": "rgba(202, 138, 4, 0.55)",
        "edge_unknown": "rgba(120, 53, 15, 0.25)",
        "node_border": "#fff7ed", "highlight_bg": "#dc2626", "hover_bg": "#f59e0b",
        "shadow_color": "rgba(124, 45, 18, 0.15)", "plotly_bg": "#fff7ed", "plotly_paper": "#fff7ed",
        "grid_color": "#fed7aa", "axis_color": "#9a3412"
    },
    "Forest": {
        "bg": "#f0fdf4", "font": "#052e16", "tooltip_bg": "rgba(240, 253, 244, 0.97)",
        "tooltip_border": "#86efac", "tooltip_text": "#052e16",
        "edge_cooccurrence": "rgba(22, 163, 74, 0.45)",
        "edge_semantic": "rgba(5, 150, 105, 0.40)",
        "edge_bridge": "rgba(234, 179, 8, 0.55)",
        "edge_unknown": "rgba(20, 83, 45, 0.25)",
        "node_border": "#f0fdf4", "highlight_bg": "#15803d", "hover_bg": "#84cc16",
        "shadow_color": "rgba(20, 83, 45, 0.15)", "plotly_bg": "#f0fdf4", "plotly_paper": "#f0fdf4",
        "grid_color": "#bbf7d0", "axis_color": "#166534"
    },
    "Ocean": {
        "bg": "#ecfeff", "font": "#083344", "tooltip_bg": "rgba(236, 254, 255, 0.97)",
        "tooltip_border": "#67e8f9", "tooltip_text": "#083344",
        "edge_cooccurrence": "rgba(6, 182, 212, 0.45)",
        "edge_semantic": "rgba(14, 165, 233, 0.40)",
        "edge_bridge": "rgba(99, 102, 241, 0.55)",
        "edge_unknown": "rgba(21, 94, 117, 0.25)",
        "node_border": "#ecfeff", "highlight_bg": "#0ea5e9", "hover_bg": "#22d3ee",
        "shadow_color": "rgba(8, 51, 68, 0.15)", "plotly_bg": "#ecfeff", "plotly_paper": "#ecfeff",
        "grid_color": "#a5f3fc", "axis_color": "#0e7490"
    }
}

PHYSICS_PRESETS = {
    "Stable (Default)": {
        "damping": 0.55, "gravity": -2500, "spring_length": 140,
        "spring_strength": 0.05, "central_gravity": 0.25, "stabilization": 2500
    },
    "Fluid": {
        "damping": 0.25, "gravity": -1800, "spring_length": 120,
        "spring_strength": 0.05, "central_gravity": 0.30, "stabilization": 1500
    },
    "Tight": {
        "damping": 0.70, "gravity": -4000, "spring_length": 80,
        "spring_strength": 0.08, "central_gravity": 0.20, "stabilization": 3000
    },
    "Readability (Spaced)": {
        "damping": 0.60, "gravity": -3500, "spring_length": 220,
        "spring_strength": 0.04, "central_gravity": 0.15, "stabilization": 3000
    },
    "Off": {
        "damping": 0.99, "gravity": 0, "spring_length": 200,
        "spring_strength": 0.0, "central_gravity": 0.0, "stabilization": 0
    }
}


# ==========================================
# v3.1: NODE CENTERING & LAYOUT OPTIONS
# ==========================================
def compute_node_layout(nx_graph, layout_mode="force-directed", weight_attr="weight"):
    """
    v3.1: Compute node positions with multiple layout strategies.

    Args:
        nx_graph: NetworkX graph
        layout_mode: "force-directed" (default), "hierarchical", "circular", "spring", "kamada_kawai"
        weight_attr: Edge attribute to use for weighting

    Returns:
        dict mapping node -> (x, y) position in normalized coordinates [-1, 1]
    """
    n_nodes = len(nx_graph.nodes())
    if n_nodes == 0:
        return {}

    if layout_mode == "force-directed" or layout_mode == "spring":
        k = max(2.5, 15.0 / np.sqrt(n_nodes + 1))
        pos = nx.spring_layout(nx_graph, k=k, iterations=200, seed=42, weight=weight_attr)

    elif layout_mode == "kamada_kawai":
        if n_nodes < 300:
            pos = nx.kamada_kawai_layout(nx_graph, weight=weight_attr)
        else:
            pos = nx.spring_layout(nx_graph, k=2.5, iterations=200, seed=42, weight=weight_attr)

    elif layout_mode == "hierarchical":
        type_to_level = {
            "material": 0,
            "process": 1,
            "microstructure": 2,
            "defect": 2,
            "property": 3,
            "method": 4,
            "characterization": 4,
            "computational": 4,
            "functional": 5,
            "general": 3
        }
        level_groups = defaultdict(list)
        for node in nx_graph.nodes():
            concept_type = nx_graph.nodes[node].get("concept_type", "general")
            level = type_to_level.get(concept_type, 3)
            level_groups[level].append(node)

        pos = {}
        max_level = max(level_groups.keys()) if level_groups else 0

        for level, nodes in sorted(level_groups.items()):
            radius = 0.2 + 0.8 * (level / max(max_level, 1))
            n_in_level = len(nodes)
            for i, node in enumerate(nodes):
                angle = 2 * np.pi * i / max(n_in_level, 1) - np.pi / 2
                pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

        if n_nodes > 5:
            spring_pos = nx.spring_layout(nx_graph, k=0.5, iterations=50, seed=42, weight=weight_attr)
            for node in pos:
                if node in spring_pos:
                    pos[node] = (0.7 * pos[node][0] + 0.3 * spring_pos[node][0],
                                0.7 * pos[node][1] + 0.3 * spring_pos[node][1])

    elif layout_mode == "circular":
        nodes_by_degree = sorted(nx_graph.nodes(), 
                                key=lambda n: nx_graph.degree(n, weight=weight_attr), 
                                reverse=True)
        pos = {}
        for i, node in enumerate(nodes_by_degree):
            angle = 2 * np.pi * i / max(n_nodes, 1) - np.pi / 2
            pos[node] = (np.cos(angle), np.sin(angle))

    else:
        pos = nx.spring_layout(nx_graph, k=2.5, iterations=200, seed=42, weight=weight_attr)

    return pos


# ==========================================
# v3.1: ENHANCED PYVIS RENDERER
# ==========================================
def render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=True,
                        min_node_size=8, max_node_size=40, cmap_name="viridis",
                        custom_labels=None, node_label_size=12, top_n_nodes=0,
                        theme=None, physics_preset=None,
                        use_abbreviated_labels=False, max_label_length=15,
                        edge_label_mode="hover", layout_mode="force-directed"):
    """
    v3.1: Publication-ready PyVis with:
    - Multiple layout modes (force-directed, hierarchical, circular)
    - N1, N2... abbreviated labels inside circle nodes
    - HTML legend mapping abbreviations to full names
    - Edge value inspection (weight, type, confidence, inference status on hover/tooltips)
    - Dynamic edge opacity based on weight
    - Enhanced node centering via layout_mode parameter
    """
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight="weight"))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if physics_preset is None:
        physics_preset = PHYSICS_PRESETS["Stable (Default)"]

    pos = compute_node_layout(nx_graph, layout_mode=layout_mode, weight_attr="weight")

    for node in pos:
        pos[node] = (pos[node][0] * 1200, pos[node][1] * 1200)

    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))

    net = Network(
        height="780px", width="100%", bgcolor=theme["bg"], font_color=theme["font"],
        select_menu=True, notebook=False, cdn_resources="remote"
    )

    if physics_enabled and physics_preset.get("gravity", 0) != 0:
        net.set_options(f"""
        var options = {{
          "physics": {{
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {{
              "gravitationalConstant": {physics_preset["gravity"]},
              "centralGravity": {physics_preset["central_gravity"]},
              "springLength": {physics_preset["spring_length"]},
              "springConstant": {physics_preset["spring_strength"]},
              "damping": {physics_preset["damping"]},
              "overlap": 0.15
            }},
            "stabilization": {{
              "enabled": true,
              "iterations": {physics_preset["stabilization"]},
              "updateInterval": 30,
              "onlyDynamicEdges": false,
              "fit": true
            }}
          }},
          "interaction": {{
            "hover": true,
            "tooltipDelay": 180,
            "hideEdgesOnDrag": false,
            "zoomView": true,
            "dragView": true
          }}
        }}
        """)
    else:
        net.set_options("""
        var options = {
          "physics": { "enabled": false },
          "interaction": { "hover": true, "dragNodes": true, "dragView": true, "zoomView": true }
        }
        """)

    CATEGORY_BORDER_COLORS = {
        "material": "#D32F2F",
        "process": "#00BCD4",
        "property": "#4CAF50",
        "microstructure": "#FF9800",
        "characterization": "#3F51B5",
        "computational": "#8E24AA",
        "functional": "#FFC107",
        "defect_engineering": "#E65100",
        "general": "#9E9E9E"
    }

    label_map = {}
    n_counter = 1

    for i, node in enumerate(nx_graph.nodes()):
        original_label = node
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(min_node_size + freq * 1.2, min_node_size, max_node_size))
        color = get_nanomaterials_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        category = get_concept_category(node)
        border_color = CATEGORY_BORDER_COLORS.get(category, "#9E9E9E")
        concept_type = nx_graph.nodes[node].get("concept_type", "general")

        x, y = pos.get(node, (0, 0))

        if use_abbreviated_labels and len(original_label) > max_label_length:
            short_label = f"N{n_counter}"
            label_map[short_label] = original_label
            n_counter += 1
            display_label = short_label
            node_shape = "circle"
            inside_font_size = max(8, min(int(size * 0.55), 14))
            font_color = "#ffffff"
            font_vadjust = 0
            font_align = "center"
        else:
            display_label = custom_labels.get(node, node) if custom_labels else node
            node_shape = "dot"
            inside_font_size = node_label_size
            font_color = theme["font"]
            font_vadjust = -6
            font_align = "left"

        tooltip_html = f"""<div style="font-family:Inter,sans-serif;">
        <b style="font-size:14px;color:{theme["highlight_bg"]};">{original_label}</b><br>
        <span style="color:{theme["tooltip_text"]};opacity:0.7;">Abbreviation:</span> {short_label if use_abbreviated_labels and len(original_label) > max_label_length else "N/A"}<br>
        <span style="color:{theme["tooltip_text"]};opacity:0.7;">Category:</span> {category}<br>
        <span style="color:{theme["tooltip_text"]};opacity:0.7;">Type:</span> {concept_type}<br>
        <span style="color:{theme["tooltip_text"]};opacity:0.7;">Degree:</span> {degree}<br>
        <span style="color:{theme["tooltip_text"]};opacity:0.7;">Frequency:</span> {freq}
        </div>"""

        net.add_node(
            node,
            label=display_label,
            size=size,
            x=x,
            y=y,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": theme["highlight_bg"], "border": "#ffffff"},
                "hover": {"background": theme["hover_bg"], "border": "#ffffff"}
            },
            font={
                "color": font_color,
                "size": inside_font_size,
                "face": "Inter, Segoe UI, Roboto, sans-serif",
                "strokeWidth": 0,
                "vadjust": font_vadjust,
                "align": font_align
            },
            title=tooltip_html,
            borderWidth=3,
            borderWidthSelected=4,
            shadow={
                "enabled": True,
                "color": theme["shadow_color"],
                "size": 12,
                "x": 4,
                "y": 4
            },
            shape=node_shape,
            mass=max(1, 1 + freq * 0.05)
        )

    all_weights = [nx_graph[u][v].get("weight", 1) for u, v in nx_graph.edges()]
    max_weight = max(all_weights) if all_weights else 1.0
    weight_threshold = np.percentile(all_weights, 80) if all_weights else 0

    color_map = {
        "cooccurrence": theme["edge_cooccurrence"],
        "semantic": theme["edge_semantic"],
        "bridge": theme["edge_bridge"],
        "hypernym": "rgba(156, 39, 176, 0.5)",
        "hyponym": "rgba(156, 39, 176, 0.3)",
        "causes": "rgba(244, 67, 54, 0.6)",
        "influences": "rgba(255, 152, 0, 0.6)",
        "unknown": theme["edge_unknown"]
    }

    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get("weight", 1)
        edge_type = nx_graph[u][v].get("edge_type", "unknown")
        is_inferred = nx_graph[u][v].get("inferred", False)
        confidence = nx_graph[u][v].get("confidence", 1.0)
        evidence = nx_graph[u][v].get("evidence", "")
        path = nx_graph[u][v].get("path", "")

        color = color_map.get(edge_type, color_map["unknown"])
        opacity = float(np.clip(0.3 + (w / max_weight) * 0.7, 0.3, 1.0))
        width = float(np.clip(w * 0.5, 1.0, 4.0))

        edge_tooltip = f"""<span style="font-family:Inter,sans-serif;">
        <b>{u} ↔ {v}</b><br>
        Weight: <b>{w:.2f}</b><br>
        Type: <b>{edge_type}</b><br>
        Inferred: <b>{"Yes" if is_inferred else "No"}</b><br>
        Confidence: <b>{confidence:.2f}</b><br>
        {f"Path: {path}<br>" if path else ""}
        {f"Evidence: {evidence[:100]}..." if evidence else ""}
        </span>"""

        edge_kwargs = dict(
            value=float(np.clip(w, 0.5, 5)),
            width=width,
            color={
                "color": color,
                "highlight": theme["highlight_bg"],
                "hover": theme["hover_bg"],
                "opacity": opacity
            },
            smooth={"type": "continuous", "roundness": 0.5},
            dashes=True if is_inferred else False,
            title=edge_tooltip
        )

        actual_edge_label_color = theme["font"]
        if edge_label_mode == "all":
            edge_kwargs["label"] = f"{w:.1f}"
            edge_kwargs["font"] = {
                "color": actual_edge_label_color,
                "size": 10,
                "background": theme["tooltip_bg"],
                "strokeWidth": 2,
                "strokeColor": theme["node_border"],
                "align": "middle",
                "face": "Inter, Segoe UI, Roboto, sans-serif"
            }
        elif edge_label_mode == "threshold" and w >= weight_threshold:
            edge_kwargs["label"] = f"{w:.1f}"
            edge_kwargs["font"] = {
                "color": actual_edge_label_color,
                "size": 10,
                "background": theme["tooltip_bg"],
                "strokeWidth": 2,
                "strokeColor": theme["node_border"],
                "align": "middle",
                "face": "Inter, Segoe UI, Roboto, sans-serif"
            }

        net.add_edge(u, v, **edge_kwargs)

    html_content = net.generate_html()

    custom_css = f"""
    <style>
        body {{
            background: {theme["bg"]};
            margin: 0;
            padding: 0;
            font-family: "Inter", "Segoe UI", sans-serif;
        }}
        #mynetwork {{
            border-radius: 16px;
            box-shadow: 0 12px 48px {theme["shadow_color"]};
            outline: none;
        }}
        div.vis-tooltip {{
            background: {theme["tooltip_bg"]} !important;
            color: {theme["tooltip_text"]} !important;
            border: 1px solid {theme["tooltip_border"]} !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            font-family: "Inter", "Segoe UI", sans-serif !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
            box-shadow: 0 8px 32px {theme["shadow_color"]} !important;
            max-width: 320px !important;
            white-space: normal !important;
        }}
        div.vis-network div.vis-manipulation {{
            background: {theme["tooltip_bg"]} !important;
            border-top: 1px solid {theme["tooltip_border"]} !important;
            color: {theme["font"]} !important;
        }}
    </style>
    """
    html_content = html_content.replace("</head>", custom_css + "</head>")

    st.components.v1.html(html_content, height=790, scrolling=True)

    if use_abbreviated_labels and label_map:
        st.markdown("### 🗺️ Node Label Legend (N1, N2... → Full Names)")
        sorted_legend = sorted(label_map.items(), key=lambda x: int(x[0][1:]))

        cols_per_row = 4
        for i in range(0, len(sorted_legend), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, (short, full) in enumerate(sorted_legend[i:i+cols_per_row]):
                with cols[j]:
                    cat_color = get_nanomaterials_category_color(full)
                    st.markdown(f"""
                    <div style="padding:4px 8px; border-radius:4px; background:rgba(0,0,0,0.05); margin-bottom:4px;">
                        <span style="color:{cat_color}; font-weight:bold;">{short}</span>: 
                        <span style="font-size:0.85em;">{full}</span>
                    </div>
                    """, unsafe_allow_html=True)

    try:
        html_bytes = html_content.encode("utf-8")
        st.download_button("📥 Download Interactive Graph (HTML)", data=html_bytes,
                          file_name="nanomaterials_concept_graph_v3.html", mime="text/html")
        del html_content, html_bytes
        import gc
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")


# ==========================================
# v3.1: ENHANCED SUNBURST CHART
# ==========================================
def _build_path(label, parent_map):
    """Helper: Build full path from root to label."""
    path = []
    curr = label
    visited = set()
    while curr != "" and curr not in visited:
        visited.add(curr)
        path.insert(0, curr)
        curr = parent_map.get(curr, "")
    return path


def render_sunburst_chart(labels, parents, values, cmap_name="viridis", 
                            label_size=11, width=800, height=600, theme=None,
                            use_symbols=True):
    """
    v3.1: Dramatically enhanced sunburst with symbol-based hierarchical representation.

    Key improvements over v3.0:
    - insidetextorientation="horizontal" keeps symbols perfectly upright
    - Spaced symbols ("✦ ★ ●" instead of "✦★●") for readability
    - customdata + hovertemplate fixes: hover shows ACTUAL concept name, not symbols
    - Bold font weight (700) and larger size for symbol visibility
    - Enhanced slice borders (1.5px white) for visual separation
    - Adaptive text color (white on dark slices, dark on light slices)
    - Symbol legend with depth-grouped proportional bars
    """
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return

    n_items = len(labels)
    use_remainder = n_items > 80

    unique_ids = []
    seen = {}
    for i, lab in enumerate(labels):
        base = lab[:25] + ("..." if len(lab) > 25 else "")
        if base in seen:
            unique_ids.append(f"{base}_{seen[base]}")
            seen[base] += 1
        else:
            unique_ids.append(base)
            seen[base] = 1

    parent_ids = []
    for p in parents:
        if p == "":
            parent_ids.append("")
        else:
            for i, lab in enumerate(labels):
                if lab == p:
                    parent_ids.append(unique_ids[i])
                    break
            else:
                parent_ids.append("")

    SYMBOL_LIBRARY = ["✦", "★", "●", "■", "▲", "◆", "⬟", "⬢", "◉", "◈", "◇", "○"]

    parent_map = {labels[i]: parents[i] for i in range(len(labels))}

    def get_depth(label):
        depth = 0
        curr = label
        visited = set()
        while parent_map.get(curr, "") != "" and curr not in visited:
            visited.add(curr)
            curr = parent_map[curr]
            depth += 1
        return depth

    depths = [get_depth(l) for l in labels]

    node_symbols = {}
    display_labels = []

    if use_symbols:
        for i, lab in enumerate(labels):
            d = depths[i]
            if d == 0:
                node_symbols[lab] = SYMBOL_LIBRARY[0]
                display_labels.append(SYMBOL_LIBRARY[0])
            else:
                chain = []
                curr = lab
                visited = set()
                while curr != "" and curr not in visited:
                    visited.add(curr)
                    if curr in node_symbols:
                        chain.insert(0, node_symbols[curr])
                    curr = parent_map.get(curr, "")

                siblings = [labels[j] for j in range(len(labels)) 
                           if parents[j] == parents[i] and depths[j] == d]
                sym_idx = (d + siblings.index(lab)) % len(SYMBOL_LIBRARY)
                own_symbol = SYMBOL_LIBRARY[sym_idx]
                node_symbols[lab] = own_symbol
                chain.append(own_symbol)

                display_labels.append("  ".join(chain[-3:]))
    else:
        display_labels = unique_ids

    try:
        if cmap_name in ["turbo", "cividis", "viridis", "plasma", "inferno"]:
            cmap_obj = matplotlib.colormaps.get_cmap(cmap_name)
        else:
            cmap_obj = matplotlib.colormaps.get_cmap("turbo")
        t_vals = np.linspace(0.05, 0.95, len(unique_ids))
        rgbas = [cmap_obj(t) for t in t_vals]
        plot_colors = [matplotlib.colors.to_hex(rgba) for rgba in rgbas]
    except Exception:
        plot_colors = get_colormap_colors(cmap_name, len(unique_ids))

    def hex_to_luminance(hex_color):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    text_colors = []
    for color in plot_colors:
        lum = hex_to_luminance(color)
        text_colors.append("#ffffff" if lum < 0.5 else "#1a1a1a")

    branchvalues = "remainder" if use_remainder else "total"

    customdata_list = []
    for lab in labels:
        path_str = " → ".join(_build_path(lab, parent_map))
        customdata_list.append([lab, path_str])

    fig = go.Figure(go.Sunburst(
        ids=unique_ids,
        labels=display_labels,
        parents=parent_ids,
        values=values,
        branchvalues=branchvalues,
        customdata=customdata_list,
        marker=dict(
            colors=plot_colors,
            line=dict(width=1.5, color="rgba(255,255,255,0.8)")
        ),
        textinfo="label",
        insidetextorientation="horizontal",
        insidetextfont=dict(
            size=int(label_size) + 2,
            family="Arial Black, Arial, sans-serif",
            weight=700,
            color=text_colors
        ),
        hovertemplate=(
            "<b style=\\\"font-size:14px\\\">%{customdata[0]}</b><br>"
            "<span style=\\\"color:#888\\\">Path: %{customdata[1]}</span><br>"
            "Value: <b>%{value}</b><br>"
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#cbd5e1",
            font=dict(family="Inter, sans-serif", size=13, color="#1e293b")
        )
    ))

    fig.update_layout(
        title=dict(
            text="<b>Nanomaterials Research Domain Hierarchy</b><br><i>Size = concept frequency • Hover for details</i>",
            font=dict(size=16, family="Arial, sans-serif")
        ),
        font=dict(size=label_size, family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=width,
        height=height,
        margin=dict(t=80, b=20, l=20, r=20),
        sunburstcolorway=plot_colors
    )

    st.plotly_chart(fig, use_container_width=True)

    if use_symbols:
        st.markdown("### 📊 Symbol-to-Label Legend")
        st.caption("💡 **Tip:** Click on any category slice to drill down. Symbol chains show hierarchical depth (e.g., ★ ● = category → sub-concept).")

        legend_entries = []
        for i, lab in enumerate(labels):
            d = depths[i]
            sym = display_labels[i] if use_symbols else lab
            color = plot_colors[i] if i < len(plot_colors) else "#9E9E9E"
            legend_entries.append({
                "symbol": sym,
                "label": lab,
                "depth": d,
                "color": color,
                "value": values[i]
            })
        legend_entries.sort(key=lambda x: (x["depth"], -x["value"]))

        depth_names = {
            0: "🌐 Root Domain",
            1: "📁 Major Category",
            2: "📄 Sub-category",
            3: "🔍 Concept Group",
            4: "📎 Specific Concept",
            5: "📌 Detailed Term"
        }

        current_depth = -1
        for entry in legend_entries:
            if entry["depth"] != current_depth:
                current_depth = entry["depth"]
                depth_header = depth_names.get(current_depth, f"Level {current_depth}")
                st.markdown(f"**{depth_header}**")

            same_depth_vals = [e["value"] for e in legend_entries if e["depth"] == current_depth]
            max_val = max(same_depth_vals) if same_depth_vals else 1
            bar_width_pct = (entry["value"] / max_val * 100) if max_val > 0 else 0

            st.markdown(
                f"""<div style="padding:6px 10px; border-radius:6px;
                background: linear-gradient(90deg, {entry["color"]}22 {bar_width_pct}%, transparent {bar_width_pct}%);
                border-left:4px solid {entry["color"]}; margin-bottom:4px;
                display: flex; align-items: center;">
                <span style="font-size:20px; margin-right:10px;">{entry["symbol"]}</span>
                <span style="font-size:13px; color:#333; font-weight:500; flex-grow:1;">{entry["label"]}</span>
                <span style="font-size:11px; color:#666;"> ({entry["value"]})</span>
                </div>""",
                unsafe_allow_html=True
            )


# ==========================================
# v3.1: CATEGORY COLOR HELPERS
# ==========================================
def get_nanomaterials_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    if any(c in concept_lower for c in ["nanotwinned cu", "nt cu", "nanotwin"]):
        return "#D32F2F"
    elif any(c in concept_lower for c in ["cu@ag", "cu/ag", "core shell cu", "core shell ag", "bimetallic cu", "bimetallic ag"]):
        return "#1976D2"
    elif any(c in concept_lower for c in ["defect engineered ag", "defect engineered silver", "ag defect", "silver defect", "vacancy ag"]):
        return "#388E3C"
    elif any(c in concept_lower for c in ["defect_engineering", "defect_design", "defect_tailoring"]):
        return "#E65100"
    elif any(c in concept_lower for c in ["irradiation_damage", "radiation_defect", "ion_implantation"]):
        return "#BF360C"
    elif any(c in concept_lower for c in ["vacancy", "interstitial", "frenkel", "schottky", "point_defect"]):
        return "#6A1B9A"
    elif any(c in concept_lower for c in ["dislocation_loop", "dislocation_dipole", "dislocation_jog", "dislocation_substructure"]):
        return "#AD1457"
    elif any(c in concept_lower for c in ["void", "cavity", "pore", "bubble"]):
        return "#455A64"
    elif any(c in concept_lower for c in ["precipitate", "gp_zone", "guinier", "second_phase"]):
        return "#00796B"
    elif any(c in concept_lower for c in ["twin boundary", "twin density", "twin spacing", "ctb", "itb"]):
        return "#E91E63"
    elif any(c in concept_lower for c in ["stacking fault", "sfe", "gsfe", "shockley", "frank partial"]):
        return "#9C27B0"
    elif any(c in concept_lower for c in ["dislocation", "burgers", "slip system", "cross slip", "gnd", "ssd"]):
        return "#FF9800"
    elif any(c in concept_lower for c in ["grain boundary", "grain size", "triple junction", "csl"]):
        return "#795548"
    elif any(c in concept_lower for c in ["yield strength", "uts", "hardness", "elastic modulus", "young modulus", "ductility", "elongation"]):
        return "#F44336"
    elif any(c in concept_lower for c in ["fracture", "fatigue", "creep", "wear", "damage"]):
        return "#FF5722"
    elif any(c in concept_lower for c in ["electrodeposition", "sputtering", "cvd", "ald", "pld", "mbe", "evaporation"]):
        return "#00BCD4"
    elif any(c in concept_lower for c in ["ecap", "hpt", "arb", "fsp", "spd", "ball milling", "cryorolling"]):
        return "#009688"
    elif any(c in concept_lower for c in ["annealing", "recrystallization", "grain growth", "thermal"]):
        return "#8BC34A"
    elif any(c in concept_lower for c in ["tem", "hrtem", "stem", "ebsd", "xrd", "apt", "synchrotron"]):
        return "#3F51B5"
    elif any(c in concept_lower for c in ["dft", "molecular dynamics", "md", "phase field", "finite element", "ddd"]):
        return "#4CAF50"
    elif any(c in concept_lower for c in ["machine learning", "neural network", "gnn", "cnn", "rnn", "transformer"]):
        return "#8E24AA"
    elif any(c in concept_lower for c in ["electrical conductivity", "resistivity", "thermal conductivity", "thermal expansion"]):
        return "#FFC107"
    elif any(c in concept_lower for c in ["corrosion", "oxidation", "electrochemical"]):
        return "#FFEB3B"
    elif any(c in concept_lower for c in ["catalytic", "electrocatalysis", "her", "oer", "orr", "plasmon", "spr", "lspr", "sers"]):
        return "#00E676"
    elif any(c in concept_lower for c in ["interconnect", "tsv", "solder", "conductive ink", "flexible", "transparent"]):
        return "#18FFFF"
    else:
        return "#9E9E9E"


def get_concept_category(concept: str) -> str:
    concept_lower = concept.lower()
    if any(c in concept_lower for c in ["nanotwinned cu", "nt cu", "nanotwin", "cu@ag", "cu/ag", "core shell", "defect engineered ag", "ag defect"]):
        return "material"
    elif any(c in concept_lower for c in ["twin boundary", "stacking fault", "dislocation", "grain boundary", "vacancy", "point defect", "void", "precipitate"]):
        return "microstructure"
    elif any(c in concept_lower for c in ["yield strength", "uts", "hardness", "elastic modulus", "ductility", "fracture", "fatigue"]):
        return "property"
    elif any(c in concept_lower for c in ["electrodeposition", "sputtering", "cvd", "ecap", "hpt", "annealing", "recrystallization"]):
        return "process"
    elif any(c in concept_lower for c in ["tem", "hrtem", "ebsd", "xrd", "apt"]):
        return "characterization"
    elif any(c in concept_lower for c in ["dft", "molecular dynamics", "machine learning", "gnn"]):
        return "computational"
    elif any(c in concept_lower for c in ["electrical conductivity", "corrosion", "catalytic", "interconnect"]):
        return "functional"
    elif any(c in concept_lower for c in ["defect_engineering", "irradiation", "defect_design", "defect_tailoring"]):
        return "defect_engineering"
    else:
        return "general"


# ==========================================
# v3.1: ENHANCED SIDEBAR WITH LAYOUT MODE
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🎨 Theme")
        st.session_state["theme"] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0
        )
        theme = THEME_PRESETS[st.session_state["theme"]]

        st.subheader("🔬 Nanomaterials Focus Areas")
        st.markdown("- Nanotwinned Cu (twin boundaries, CTB/ITB)")
        st.markdown("- Core-shell Cu@Ag nanoparticles (interface engineering)")
        st.markdown("- Defect-engineered Ag nanoparticles (vacancies, dislocations)")
        st.markdown("- Defect Engineering (irradiation, vacancy control, dislocation design)")
        st.markdown("- Mechanical properties (strength, ductility, hardness)")
        st.markdown("- Synthesis (electrodeposition, SPD, annealing)")
        st.markdown("- Characterization (TEM, EBSD, XRD, APT)")
        st.markdown("- Computational methods (DFT, MD, ML potentials)")

        st.subheader("🖼️ Visualization")
        st.session_state["viz_backend"] = st.selectbox(
            "Engine:", ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"], index=0
        )
        st.session_state["cmap_name"] = st.selectbox(
            "Colormap:", options=list(SUPPORTED_COLORMAPS.keys()), index=0
        )

        st.subheader("🔧 Physics & Layout")
        st.session_state["layout_mode"] = st.selectbox(
            "Layout mode:",
            ["force-directed", "hierarchical", "circular", "kamada_kawai"],
            index=0,
            help="force-directed: Standard physics-based layout\nhierarchical: Groups by concept type in concentric rings\ncircular: Arranges nodes on a circle by degree\nkamada_kawai: Energy-minimizing (best for <300 nodes)"
        )

        st.session_state["physics_preset"] = st.selectbox(
            "Physics preset:",
            options=list(PHYSICS_PRESETS.keys()),
            index=0
        )
        preset = PHYSICS_PRESETS[st.session_state["physics_preset"]]
        st.session_state["physics_enabled"] = st.checkbox(
            "Enable physics", value=(preset["gravity"] != 0)
        )

        with st.expander("⚙️ Advanced Physics Overrides"):
            st.session_state["adv_damping"] = st.slider("Damping", 0.05, 0.95, preset["damping"], step=0.05)
            st.session_state["adv_gravity"] = st.slider("Repulsion", -8000, -500, preset["gravity"], step=100)
            st.session_state["adv_spring_length"] = st.slider("Spring length", 40, 300, preset["spring_length"], step=10)
            st.session_state["adv_spring_strength"] = st.slider("Spring strength", 0.01, 0.20, preset["spring_strength"], step=0.01)
            st.session_state["adv_central_gravity"] = st.slider("Central gravity", 0.0, 0.5, preset["central_gravity"], step=0.05)
            st.session_state["adv_stabilization"] = st.slider("Stabilization iter", 0, 5000, preset["stabilization"], step=250)

        base_preset = PHYSICS_PRESETS[st.session_state["physics_preset"]].copy()
        if st.session_state.get("adv_damping") is not None:
            base_preset["damping"] = st.session_state["adv_damping"]
            base_preset["gravity"] = st.session_state["adv_gravity"]
            base_preset["spring_length"] = st.session_state["adv_spring_length"]
            base_preset["spring_strength"] = st.session_state["adv_spring_strength"]
            base_preset["central_gravity"] = st.session_state["adv_central_gravity"]
            base_preset["stabilization"] = st.session_state["adv_stabilization"]
        st.session_state["effective_physics"] = base_preset

        st.subheader("📊 Display Limits")
        col_all1, col_slider1 = st.columns([0.3, 0.7])
        with col_all1:
            all_graph = st.checkbox("All", value=True, key="all_graph_chk")
        with col_slider1:
            st.session_state["top_n_graph"] = st.slider(
                "Max nodes", 10, 500, 200, step=10, disabled=all_graph,
                key="top_n_graph_slider"
            )
        if all_graph:
            st.session_state["top_n_graph"] = 0

        col_all2, col_slider2 = st.columns([0.3, 0.7])
        with col_all2:
            all_sun = st.checkbox("All", value=True, key="all_sun_chk")
        with col_slider2:
            st.session_state["top_n_sunburst"] = st.slider(
                "Max children/category", 10, 100, 40, step=10, disabled=all_sun,
                key="top_n_sunburst_slider"
            )
        if all_sun:
            st.session_state["top_n_sunburst"] = 0

        col_all3, col_slider3 = st.columns([0.3, 0.7])
        with col_all3:
            all_radar = st.checkbox("All", value=True, key="all_radar_chk")
        with col_slider3:
            st.session_state["top_n_radar"] = st.slider(
                "Top K for radar", 5, 30, 15, disabled=all_radar,
                key="top_n_radar_slider"
            )
        if all_radar:
            st.session_state["top_n_radar"] = 0

        st.subheader("🔧 Graph Parameters")
        st.session_state["min_freq"] = st.slider("Min concept frequency", 1, 20, 1)
        st.session_state["min_words"] = st.slider("Min words per concept", 2, 5, 2)
        st.session_state["sim_threshold"] = st.slider("Semantic threshold", 0.6, 0.95, 0.85, step=0.05)
        st.session_state["cooc_weight"] = st.slider("Co-occurrence weight", 0.5, 1.0, 0.9, step=0.1)
        st.session_state["sem_weight"] = st.slider("Semantic weight", 0.0, 0.5, 0.1, step=0.1)

        st.subheader("🧠 Reasoning Engine")
        st.session_state["use_global_resolution"] = st.checkbox(
            "🚀 Use global batch resolution (10-50x faster)", value=True
        )
        st.session_state["use_ontology"] = st.checkbox(
            "📚 Enable ontology resolution", value=True
        )
        st.session_state["use_context_disambiguation"] = st.checkbox(
            "🔍 Enable context disambiguation", value=True
        )
        st.session_state["use_causal_extraction"] = st.checkbox(
            "🔗 Enable cause-effect extraction", value=True
        )
        st.session_state["use_inference"] = st.checkbox(
            "🌉 Enable cross-domain bridge inference", value=True
        )
        st.session_state["parallel_workers"] = st.slider(
            "Parallel workers", 1, 16, 8, step=1
        )

        st.subheader("🎨 Visualization Options")
        st.session_state["use_abbreviated_labels"] = st.checkbox(
            "🏷️ Use N1/N2 abbreviated labels", value=False
        )
        st.session_state["max_label_length"] = st.slider(
            "Max label length before abbreviation", 2, 50, 15, step=1
        )
        st.session_state["edge_label_mode"] = st.selectbox(
            "Edge label mode:", ["hover", "threshold", "all", "none"]
        )
        st.session_state["use_symbol_sunburst"] = st.checkbox(
            "🔣 Use symbol-based sunburst", value=True
        )

        st.subheader("📐 Statistics")
        st.session_state["bootstrap_samples"] = st.slider("Bootstrap samples", 100, 2000, 500, step=100)
        st.session_state["alpha_level"] = st.selectbox("Significance α", [0.01, 0.05, 0.10], index=1)

        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            import gc
            gc.collect()
            st.success("Cache cleared!")
        import torch
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"🖥️ Device: {gpu_info}")


# ==========================================
# v3.1: MAIN APPLICATION
# ==========================================
def main():
    st.title("🔬 Nanomaterials-ConceptGraph v3.1: Deep Semantic Reasoning Engine")
    st.caption("Enhanced Sunburst • Hierarchical Layouts • Adaptive Contrast • Symbol Legends")
    render_sidebar()

    st.info("This is the v3.1 visualization enhancement module. Integrate these functions into your main v3.0 application.")

    st.markdown("""
    ### 📋 Integration Instructions

    1. **Replace** `render_graph_pyvis()` in your v3.0 code with the v3.1 version above
    2. **Replace** `render_sunburst_chart()` in your v3.0 code with the v3.1 version above  
    3. **Replace** `render_sidebar()` in your v3.0 code with the v3.1 version above
    4. **Add** `compute_node_layout()` as a new function
    5. **Add** `_build_path()` as a new helper function
    6. **Update** the `viz_tab` call in `main()` to pass `layout_mode`:

    ```python
    layout_mode = st.session_state.get("layout_mode", "force-directed")
    render_graph_pyvis(
        nx_graph, concept_abstract_map, ...,
        layout_mode=layout_mode  # <-- ADD THIS
    )
    ```
    """)


if __name__ == "__main__":
    main()
