#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nanomaterials-ConceptGraph: Core-Shell Ag-Cu Explorer (Enhanced)
================================================================
Added features:
- Interactive graph editing (add/remove/merge/rename nodes/edges)
- Enhanced sunburst with category filters and drill-down
- Timeline, heatmap, t-SNE, community detection, growth scores, bubble chart
- All new figures are integrated into the Streamlit UI.
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import torch.optim as optim
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import os
import sys
import tempfile
import warnings
import traceback
import gc
import hashlib
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import davies_bouldin_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches as mpatches
import seaborn as sns

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Nanomaterials-ConceptGraph: Core-Shell Ag-Cu Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PATHS & DIRECTORIES (unchanged)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)

# ==========================================
# COLORMAP REGISTRY (unchanged)
# ==========================================
SUPPORTED_COLORMAPS = { ... }  # same as before

def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    # same as before
    ...

# ==========================================
# ROBUST JSON LOADER (unchanged)
# ==========================================
def robust_load_file(filepath: Path):
    ...

@st.cache_data(show_spinner=False)
def load_all_json_files(directory):
    ...

@st.cache_data(show_spinner=False)
def build_master_dataframe(file_records):
    ...

# ==========================================
# CORE-SHELL AG-CU DOMAIN CONFIGURATION (unchanged)
# ==========================================
ALL_DOMAIN_KEYWORDS = ...  # same
NANOMATERIALS_PATTERNS = ...
NANOMATERIALS_CATEGORY_MAPPING = ...

# ==========================================
# UTILITY FUNCTIONS (unchanged)
# ==========================================
def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    ...

# ==========================================
# DEVICE & MODEL MANAGEMENT (unchanged)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    ...

# ==========================================
# CONCEPT EXTRACTION & NORMALIZATION (unchanged)
# ==========================================
def is_valid_nanomaterials_concept(concept: str) -> bool:
    ...

def normalize_nanomaterials_term(concept: str) -> str:
    ...

def extract_concepts_from_text(text: str) -> List[str]:
    ...

def extract_concepts_from_abstracts(df: pd.DataFrame, text_columns: List[str]) -> Tuple[List[List[str]], List[Dict]]:
    ...

def cluster_similar_concepts(valid_concepts: List[str], embed_model, similarity_threshold: float = 0.75):
    ...

def normalize_and_filter_concepts(all_concepts: List[List[str]], config: Dict) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
    ...

def abstract_concepts_to_categories(concepts: List[str]) -> Dict[str, str]:
    ...

# ==========================================
# CONCEPT DISTILLATION (unchanged)
# ==========================================
def compute_concept_distillation(valid_concepts: List[str], concept_abstract_map: Dict[str, List[int]], all_texts: List[str]) -> pd.DataFrame:
    ...

# ==========================================
# GRAPH CONSTRUCTION (unchanged)
# ==========================================
def build_hybrid_graph(all_concepts: List[List[str]], valid_concepts: List[str], concept_to_id: Dict[str, int], embed_model=None, config: Dict = None) -> nx.Graph:
    ...

def sample_edges_for_training(nx_graph: nx.Graph, valid_concepts: List[str], concept_to_id: Dict[str, int], config: Dict = None) -> Tuple[List[Tuple], List[Tuple]]:
    ...

# ==========================================
# GNN MODEL (unchanged)
# ==========================================
class SparseGraphSAGE(nn.Module):
    ...

def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, progress_callback=None, epochs: int = 50, lr: float = 1e-3):
    ...

# ==========================================
# RESEARCH DIRECTION SCORING (unchanged)
# ==========================================
def compute_research_direction_scores(model, node_features, final_emb, nx_graph, valid_concepts, concept_properties, ridge, embed_model, n_samples: int = 5000) -> pd.DataFrame:
    ...

# ==========================================
# MATHEMATICAL VALIDATION (unchanged)
# ==========================================
def validate_graph_metrics(nx_graph: nx.Graph, valid_concepts: List[str]) -> Dict[str, Any]:
    ...

@st.cache_data(ttl=3600)
def compute_bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05):
    ...

# ==========================================
# ============ ENHANCEMENT: GRAPH EDITING FUNCTIONS =============
# ==========================================

def apply_graph_edits(nx_graph, concept_abstract_map, valid_concepts, edits):
    """
    Apply user edits to the graph.
    edits is a dict with keys: 'remove_nodes', 'merge_nodes', 'rename_nodes', 'add_edges'
    Returns updated graph, concept_abstract_map, valid_concepts (list).
    """
    G = nx_graph.copy()
    map_abs = concept_abstract_map.copy()
    concepts = valid_concepts.copy()

    # 1. Remove nodes
    for node in edits.get('remove_nodes', []):
        if node in G:
            G.remove_node(node)
            map_abs.pop(node, None)
            if node in concepts:
                concepts.remove(node)

    # 2. Merge nodes: replace multiple nodes with a new representative name
    for merge_list, new_name in edits.get('merge_nodes', []):
        if len(merge_list) < 2:
            continue
        # Ensure all nodes exist
        if not all(n in G for n in merge_list):
            continue
        # Combine frequencies
        total_freq = sum(len(map_abs.get(n, [])) for n in merge_list)
        # Collect all abstracts
        combined_abstracts = []
        for n in merge_list:
            combined_abstracts.extend(map_abs.get(n, []))
        combined_abstracts = list(set(combined_abstracts))  # unique abstracts

        # Create new node
        G.add_node(new_name, frequency=total_freq)
        map_abs[new_name] = combined_abstracts

        # Connect new node to all neighbors of merged nodes
        neighbors = set()
        for n in merge_list:
            neighbors.update(G.neighbors(n))
        for nb in neighbors:
            if nb not in merge_list:  # avoid self-loop
                # Compute weight as sum of weights from old nodes to this neighbor
                w = 0
                for n in merge_list:
                    if G.has_edge(n, nb):
                        w += G[n][nb].get('weight', 1)
                G.add_edge(new_name, nb, weight=w, cooccurrence=w, edge_type='cooccurrence')

        # Remove old nodes
        for n in merge_list:
            G.remove_node(n)
            map_abs.pop(n, None)
            if n in concepts:
                concepts.remove(n)

        # Add new name to concepts
        if new_name not in concepts:
            concepts.append(new_name)

    # 3. Rename nodes
    for old, new in edits.get('rename_nodes', []):
        if old in G and new not in G:
            # Transfer edges and attributes
            adj = dict(G[old])
            attrs = G.nodes[old]
            G.add_node(new, **attrs)
            for nb, data in adj.items():
                G.add_edge(new, nb, **data)
            G.remove_node(old)
            # Update map
            if old in map_abs:
                map_abs[new] = map_abs.pop(old)
            # Update concepts list
            if old in concepts:
                idx = concepts.index(old)
                concepts[idx] = new

    # 4. Add edges
    for u, v, weight in edits.get('add_edges', []):
        if u in G and v in G and not G.has_edge(u, v):
            G.add_edge(u, v, weight=weight, cooccurrence=weight, edge_type='user_added')

    return G, map_abs, concepts

# ==========================================
# ============ ENHANCEMENT: NEW VISUALIZATION FUNCTIONS ============
# ==========================================

def render_timeline(df_filtered, concept_abstract_map, valid_concepts, year_col='Year'):
    """Display concept frequency over time (if Year column exists)."""
    if year_col not in df_filtered.columns:
        st.info("No 'Year' column found. Timeline unavailable.")
        return
    # Ensure Year is numeric
    df_years = df_filtered[df_filtered[year_col].notna()].copy()
    if df_years.empty:
        st.info("No valid years in data.")
        return
    df_years[year_col] = pd.to_numeric(df_years[year_col], errors='coerce')
    df_years = df_years.dropna(subset=[year_col])
    if df_years.empty:
        return

    # For each concept, count occurrences per year
    # We need to map abstracts to years: we need year per abstract index.
    # We have df_filtered with rows; each row corresponds to an abstract index.
    # concept_abstract_map maps concept -> list of abstract indices.
    # So we can get year for each abstract index.
    year_series = df_years[year_col]
    # Build mapping abstract_idx -> year (for those that have year)
    abs_year = {i: year for i, year in year_series.items() if i < len(df_filtered)}
    # For each concept, collect years
    concept_year_counts = defaultdict(lambda: defaultdict(int))
    for concept, abs_list in concept_abstract_map.items():
        if concept not in valid_concepts:
            continue
        for idx in abs_list:
            if idx in abs_year:
                y = abs_year[idx]
                concept_year_counts[concept][y] += 1

    # Prepare data for plotting
    years = sorted(set(abs_year.values()))
    if not years:
        return
    data = []
    for concept, year_dict in concept_year_counts.items():
        for y in years:
            data.append({'concept': concept, 'year': y, 'count': year_dict.get(y, 0)})
    df_timeline = pd.DataFrame(data)
    if df_timeline.empty:
        st.info("No timeline data.")
        return

    # Allow user to select top K concepts by total frequency
    total_freq = df_timeline.groupby('concept')['count'].sum().sort_values(ascending=False)
    top_k = st.slider("Select top K concepts for timeline", 3, min(30, len(total_freq)), 10, key='timeline_k')
    top_concepts = total_freq.head(top_k).index.tolist()
    df_plot = df_timeline[df_timeline['concept'].isin(top_concepts)]

    fig = px.line(df_plot, x='year', y='count', color='concept', markers=True,
                  title='Concept Frequency Over Time (Top K)',
                  labels={'count': 'Number of Abstracts', 'year': 'Year'})
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.3))
    st.plotly_chart(fig, use_container_width=True)

def render_cooccurrence_heatmap(nx_graph, valid_concepts, concept_abstract_map, top_n=30):
    """Heatmap of co-occurrence counts for top concepts."""
    if len(valid_concepts) < 3:
        st.info("Not enough concepts for heatmap.")
        return
    # Get top concepts by frequency
    freq = {c: len(concept_abstract_map.get(c, [])) for c in valid_concepts}
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_concepts = [c for c, _ in top]
    # Build co-occurrence matrix
    matrix = np.zeros((len(top_concepts), len(top_concepts)))
    for i, u in enumerate(top_concepts):
        for j, v in enumerate(top_concepts):
            if i == j:
                matrix[i,j] = 0
            else:
                # Count abstracts that contain both
                u_abs = set(concept_abstract_map.get(u, []))
                v_abs = set(concept_abstract_map.get(v, []))
                matrix[i,j] = len(u_abs & v_abs)
    # Plot heatmap
    fig = px.imshow(matrix, x=top_concepts, y=top_concepts,
                    title='Co-occurrence Heatmap (Top Concepts)',
                    color_continuous_scale='Blues',
                    labels=dict(x='Concept', y='Concept', color='Co-occurrence'))
    fig.update_layout(xaxis=dict(tickangle=45), height=700)
    st.plotly_chart(fig, use_container_width=True)

def render_tsne_projection(valid_concepts, embed_model, concept_abstract_map, cmap_name='viridis'):
    """t-SNE projection of concept embeddings."""
    if len(valid_concepts) < 5:
        st.info("Need at least 5 concepts for t-SNE.")
        return
    st.info("Computing t-SNE (may take a few seconds)...")
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        tsne = TSNE(n_components=2, perplexity=min(30, len(valid_concepts)-1), random_state=42)
        coords = tsne.fit_transform(embeddings)
        # Get categories and frequencies
        categories = [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]
        freqs = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
        df_tsne = pd.DataFrame({
            'concept': valid_concepts,
            'x': coords[:,0], 'y': coords[:,1],
            'category': categories,
            'frequency': freqs
        })
        fig = px.scatter(df_tsne, x='x', y='y', color='category', size='frequency',
                         hover_name='concept',
                         title='t-SNE Projection of Concept Embeddings',
                         labels={'x':'t-SNE 1', 'y':'t-SNE 2'},
                         size_max=20,
                         color_discrete_sequence=get_colormap_colors(cmap_name, len(df_tsne['category'].unique())))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"t-SNE failed: {e}")

def render_community_detection(nx_graph, valid_concepts):
    """Detect communities and visualize with colors."""
    if nx_graph.number_of_nodes() < 3:
        st.info("Graph too small for community detection.")
        return
    try:
        from networkx.algorithms import community
        # Greedy modularity
        comms = list(community.greedy_modularity_communities(nx_graph))
        # Assign community labels
        comm_map = {}
        for i, comm in enumerate(comms):
            for node in comm:
                comm_map[node] = i
        # Build a new graph with community colors
        G = nx_graph.copy()
        colors = get_colormap_colors('tab20', len(comms))
        node_colors = [colors[comm_map.get(n, 0) % len(colors)] for n in G.nodes()]
        # Layout
        pos = nx.spring_layout(G, seed=42)
        # Plot with matplotlib
        fig, ax = plt.subplots(figsize=(12,10))
        nx.draw(G, pos, node_color=node_colors, with_labels=True, ax=ax,
                node_size=400, font_size=8, edge_color='gray', alpha=0.7)
        ax.set_title('Community Structure (Modularity)')
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Community detection failed: {e}")

def render_concept_growth(df_filtered, concept_abstract_map, valid_concepts, year_col='Year'):
    """Compute growth rate of concepts over time and show as bar chart."""
    if year_col not in df_filtered.columns:
        st.info("No Year column. Growth analysis unavailable.")
        return
    df_years = df_filtered[df_filtered[year_col].notna()].copy()
    if df_years.empty:
        return
    df_years[year_col] = pd.to_numeric(df_years[year_col], errors='coerce')
    df_years = df_years.dropna(subset=[year_col])
    if df_years.empty:
        return
    # Get min and max year
    min_year = df_years[year_col].min()
    max_year = df_years[year_col].max()
    if max_year - min_year < 1:
        st.info("Insufficient year range.")
        return

    # For each concept, compute frequency in first and last year
    abs_year = {i: y for i, y in df_years[year_col].items() if i < len(df_filtered)}
    concept_years = defaultdict(list)
    for concept, abs_list in concept_abstract_map.items():
        if concept not in valid_concepts:
            continue
        for idx in abs_list:
            if idx in abs_year:
                concept_years[concept].append(abs_year[idx])
    growth_data = []
    for concept, years in concept_years.items():
        if not years:
            continue
        # Count frequency per year
        year_counts = Counter(years)
        # Get frequencies near min and max
        early_count = sum(year_counts.get(y, 0) for y in range(min_year, min_year+2))
        late_count = sum(year_counts.get(y, 0) for y in range(max_year-1, max_year+1))
        if early_count == 0 and late_count == 0:
            growth = 0
        else:
            growth = (late_count - early_count) / (early_count + 1)  # prevent division by zero
        total = len(years)
        growth_data.append({'concept': concept, 'growth': growth, 'total': total})
    df_growth = pd.DataFrame(growth_data)
    if df_growth.empty:
        return
    df_growth = df_growth.sort_values('growth', ascending=False).head(30)
    fig = px.bar(df_growth, x='concept', y='growth', color='total',
                 title='Concept Growth Rate (Recent vs Early)',
                 labels={'growth': 'Growth Rate', 'total': 'Total Frequency'},
                 color_continuous_scale='RdYlGn')
    fig.update_layout(xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

def render_bubble_chart(valid_concepts, concept_abstract_map, nx_graph):
    """Bubble chart: x=degree, y=frequency, size=distillation efficiency, color=category."""
    if len(valid_concepts) < 3:
        return
    freq = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
    degree = [nx_graph.degree(c) for c in valid_concepts]
    # Compute distillation efficiency roughly (simplified)
    eff = [f * np.log1p(d) for f, d in zip(freq, degree)]
    categories = [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]
    df_bubble = pd.DataFrame({
        'concept': valid_concepts,
        'frequency': freq,
        'degree': degree,
        'efficiency': eff,
        'category': categories
    })
    fig = px.scatter(df_bubble, x='degree', y='frequency', size='efficiency',
                     color='category', hover_name='concept',
                     title='Concept Landscape: Degree vs Frequency',
                     labels={'degree':'Degree (connectivity)', 'frequency':'Abstract Frequency'},
                     size_max=30,
                     color_discrete_sequence=get_colormap_colors('viridis', len(df_bubble['category'].unique())))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# VISUALIZATION FUNCTIONS (most unchanged, but added editing integration)
# ==========================================
def get_nanomaterials_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    # same as before
    ...

def render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=True,
                        min_node_size=8, max_node_size=40, cmap_name="viridis",
                        custom_labels=None, node_label_size=12, top_n_nodes=0,
                        theme=None, physics_preset=None):
    # (unchanged except we pass the possibly edited graph)
    ...

def render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name="viridis",
                            custom_labels=None, top_n_nodes=0, node_label_size=10,
                            theme=None):
    ...

def render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name="viridis", top_n_nodes=0,
                            theme=None):
    ...

def render_graph_fallback(nx_graph, concept_abstract_map, theme=None):
    ...

# ==========================================
# SUNBURST CHART (enhanced with filters)
# ==========================================
def build_category_hierarchy(valid_concepts: List[str], concept_abstract_map: Dict, top_n_per_category: int = 40,
                             categories_filter=None):
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    category_map = abstract_concepts_to_categories(valid_concepts)
    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
        if categories_filter and category not in categories_filter:
            continue
        freq = len(concept_abstract_map.get(concept, []))
        hierarchy[category]["children"].append((concept, freq))
        hierarchy[category]["count"] += freq
    for parent in list(hierarchy.keys()):
        children = hierarchy[parent]["children"]
        if top_n_per_category > 0 and len(children) > top_n_per_category:
            children.sort(key=lambda x: x[1], reverse=True)
            children = children[:top_n_per_category]
            hierarchy[parent]["count"] = sum(cnt for _, cnt in children)
            hierarchy[parent]["children"] = children
    labels, parents, values = [], [], []
    for parent, data in hierarchy.items():
        labels.append(parent); parents.append(""); values.append(data["count"])
        for child, cnt in data["children"]:
            labels.append(child); parents.append(parent); values.append(cnt)
    return labels, parents, values

def render_sunburst_chart(labels, parents, values, cmap_name="viridis", label_size=11, width=800, height=600, theme=None,
                          branchvalues='total'):
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return
    n_items = len(labels)
    unique_ids = []; seen = {}
    for i, lab in enumerate(labels):
        base = lab[:25] + ("…" if len(lab) > 25 else "")
        if base in seen:
            unique_ids.append(f"{base}_{seen[base]}")
            seen[base] += 1
        else:
            unique_ids.append(base); seen[base] = 1
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
    colors = get_colormap_colors(cmap_name, len(unique_ids))
    fig = go.Figure(go.Sunburst(
        labels=unique_ids, parents=parent_ids, values=values, ids=unique_ids,
        branchvalues=branchvalues,
        marker=dict(colors=colors, line=dict(width=0.5, color="white")),
        textinfo="label+percent entry+value",
        insidetextorientation="radial",
        textfont=dict(size=label_size),
        hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Parent: %{parent}<extra></extra>'
    ))
    fig.update_layout(
        title="<b>Core-Shell Ag-Cu Nanostructure Domain Hierarchy</b><br><i>Size = concept frequency</i>",
        font=dict(size=label_size, family="Arial"),
        paper_bgcolor="white", plot_bgcolor="white",
        width=width, height=height,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# RADAR CHART (unchanged)
# ==========================================
def render_radar_chart(concept_scores_df: pd.DataFrame, top_k: int = 15, cmap_name: str = "viridis", theme=None):
    ...

# ==========================================
# EXPORT FUNCTIONS (unchanged)
# ==========================================
def export_graph(nx_graph, concept_abstract_map, format_type: str):
    ...

# ==========================================
# GRAPH METRICS DASHBOARD (unchanged)
# ==========================================
def compute_graph_metrics(G: nx.Graph) -> dict:
    ...

def display_metric_dashboard(metrics: dict, theme=None):
    ...

# ==========================================
# THEME CONFIGURATION (unchanged)
# ==========================================
THEME_PRESETS = { ... }
PHYSICS_PRESETS = { ... }

# ==========================================
# SIDEBAR CONFIGURATION (modified to add editing tools)
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        # (unchanged theme, focus areas, visualization, physics, display limits, graph parameters, stats)
        # ... same as before ...

        # ===== ENHANCEMENT: Graph Editing Section =====
        st.subheader("✏️ Graph Editing")
        with st.expander("Edit Graph (Nodes/Edges)"):
            st.markdown("**Remove Nodes**")
            if 'analysis_data' in st.session_state and st.session_state.analysis_data is not None:
                data = st.session_state.analysis_data
                nx_graph = data['nx_graph']
                valid_concepts = data['valid_concepts']
                node_options = valid_concepts
                remove_selected = st.multiselect("Select nodes to remove", node_options, key='remove_nodes')
                if st.button("Remove Selected Nodes", key='remove_btn'):
                    if remove_selected:
                        # Apply edit
                        edits = {'remove_nodes': remove_selected}
                        G_new, map_new, concepts_new = apply_graph_edits(nx_graph, data['concept_abstract_map'], valid_concepts, edits)
                        # Update session state
                        st.session_state.analysis_data['nx_graph'] = G_new
                        st.session_state.analysis_data['concept_abstract_map'] = map_new
                        st.session_state.analysis_data['valid_concepts'] = concepts_new
                        # Also update concept_to_id etc.
                        concept_to_id = {c: i for i, c in enumerate(concepts_new)}
                        st.session_state.analysis_data['concept_to_id'] = concept_to_id
                        st.session_state.analysis_data['id_to_concept'] = {i: c for i, c in enumerate(concepts_new)}
                        st.success(f"Removed {len(remove_selected)} nodes.")
                        st.rerun()
                    else:
                        st.warning("Select at least one node.")

            st.markdown("---")
            st.markdown("**Merge Nodes**")
            if 'analysis_data' in st.session_state:
                node_options = st.session_state.analysis_data['valid_concepts']
                merge_list = st.multiselect("Select nodes to merge", node_options, key='merge_nodes_select')
                new_name = st.text_input("New concept name", key='merge_new_name')
                if st.button("Merge Selected", key='merge_btn'):
                    if len(merge_list) >= 2 and new_name.strip():
                        edits = {'merge_nodes': [(merge_list, new_name.strip())]}
                        data = st.session_state.analysis_data
                        G_new, map_new, concepts_new = apply_graph_edits(data['nx_graph'], data['concept_abstract_map'],
                                                                         data['valid_concepts'], edits)
                        st.session_state.analysis_data['nx_graph'] = G_new
                        st.session_state.analysis_data['concept_abstract_map'] = map_new
                        st.session_state.analysis_data['valid_concepts'] = concepts_new
                        concept_to_id = {c: i for i, c in enumerate(concepts_new)}
                        st.session_state.analysis_data['concept_to_id'] = concept_to_id
                        st.session_state.analysis_data['id_to_concept'] = {i: c for i, c in enumerate(concepts_new)}
                        st.success(f"Merged {len(merge_list)} nodes into '{new_name.strip()}'.")
                        st.rerun()
                    else:
                        st.warning("Select at least 2 nodes and provide a new name.")

            st.markdown("---")
            st.markdown("**Add Edge**")
            if 'analysis_data' in st.session_state:
                nodes = st.session_state.analysis_data['valid_concepts']
                u = st.selectbox("From", nodes, key='edge_u')
                v = st.selectbox("To", nodes, key='edge_v')
                weight = st.number_input("Weight", value=1.0, min_value=0.1, key='edge_weight')
                if st.button("Add Edge", key='add_edge_btn'):
                    if u != v:
                        edits = {'add_edges': [(u, v, weight)]}
                        data = st.session_state.analysis_data
                        G_new, map_new, concepts_new = apply_graph_edits(data['nx_graph'], data['concept_abstract_map'],
                                                                         data['valid_concepts'], edits)
                        st.session_state.analysis_data['nx_graph'] = G_new
                        st.session_state.analysis_data['concept_abstract_map'] = map_new
                        st.session_state.analysis_data['valid_concepts'] = concepts_new
                        st.success(f"Edge added: {u} -- {v} (weight={weight})")
                        st.rerun()
                    else:
                        st.warning("Source and target must be different.")

        # ===== ENHANCEMENT: Sunburst filter =====
        st.subheader("☀️ Sunburst Options")
        st.session_state['sunburst_categories'] = st.multiselect(
            "Filter categories",
            options=['core_shell_structure', 'bimetallic_system', 'synthesis_method', 'interfacial_structure',
                     'morphology_dimension', 'interfacial_diffusion', 'plasmonic_optical', 'catalytic_activity',
                     'biomedical_property', 'electronic_property', 'thermal_property', 'computational_method',
                     'advanced_characterization', 'standard_characterization', 'surface_chemistry', 'application_device',
                     'general'],
            default=[],
            key='sunburst_cat_filter'
        )
        st.session_state['sunburst_branchvalues'] = st.selectbox(
            "Branch values mode", ['total', 'remainder'], index=0, key='sunburst_branch'
        )

        # (rest of sidebar unchanged)
        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"🖥️ Device: {gpu_info}")

# ==========================================
# MAIN FUNCTION (modified to add new visualizations tabs)
# ==========================================
def main():
    st.title("🔬 Nanomaterials-ConceptGraph: Core-Shell Ag-Cu Nanostructure Explorer")
    st.caption("Large-corpus concept graph builder for Ag-Cu and Cu@Ag core-shell nanostructures • Optimized for plasmonics, catalysis, and interfacial engineering")
    render_sidebar()
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = None

    # ─── LOAD JSON DATA ───
    st.header("📂 Data Loading")
    st.info(f"Place JSON files in: `{JSON_METADATA_DIR}`")
    with st.spinner("Scanning json_metadatabase..."):
        file_records = load_all_json_files(JSON_METADATA_DIR)
        df = build_master_dataframe(file_records)
    if not file_records:
        st.warning("No .json files found in the directory.")
        st.info("Please place your JSON metadata files in the `json_metadatabase/` folder.")
        return
    successful_files = [f for f in file_records if f[1]]
    if not successful_files:
        st.error("Files found but none could be parsed. Check error messages above.")
        return
    st.success(f"Loaded {len(successful_files)} file(s) • {len(df)} record(s)")
    file_names = [f[0] for f in successful_files]
    selected_files = st.multiselect("Filter by source file", file_names, default=file_names)
    if selected_files:
        df_filtered = df[df["_source_file"].isin(selected_files)].copy()
    else:
        df_filtered = df.copy()
    st.write(f"Working with **{len(df_filtered)}** records")
    with st.expander("📋 Preview Data Structure"):
        st.dataframe(df_filtered.head(5), use_container_width=True)
        st.markdown("**Available columns:**")
        st.write(list(df_filtered.columns))

    # ─── TEXT COLUMN SELECTION ───
    text_cols = [c for c in df_filtered.columns if any(k in c.lower() for k in ['abstract', 'title', 'summary', 'text', 'content', 'description'])]
    if not text_cols:
        text_cols = [c for c in df_filtered.columns if df_filtered[c].dtype == 'object']
    selected_text_cols = st.multiselect(
        "Select text columns for concept extraction:",
        options=text_cols,
        default=text_cols[:2] if len(text_cols) >= 2 else text_cols
    )
    if not selected_text_cols:
        st.error("Please select at least one text column.")
        return

    # ─── RUN ANALYSIS ───
    if st.button("🚀 Build Concept Graph", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status = st.status("🔄 Initializing analysis...", expanded=True)
        try:
            with status:
                st.write("📦 Preparing text corpus...")
                all_texts = []
                for idx, row in df_filtered.iterrows():
                    text = " ".join([str(row[col]) for col in selected_text_cols if col in row and pd.notna(row[col])])
                    all_texts.append(text)
                num_abstracts = len(all_texts)
                st.write(f"✅ Prepared {num_abstracts} documents")
                progress_bar.progress(0.05)
                st.write("🧠 Loading embedding model...")
                embed_model = load_embedding_model()
                st.success("✅ Embedding model loaded")
                progress_bar.progress(0.10)
                config = get_adaptive_config(num_abstracts)
                config["MIN_CONCEPT_FREQ"] = st.session_state.get('min_freq', 5)
                config["MIN_CONCEPT_LENGTH_WORDS"] = st.session_state.get('min_words', 2)
                config["SIMILARITY_THRESHOLD"] = st.session_state.get('sim_threshold', 0.85)
                config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.9)
                config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.1)
                st.write(f"📊 Adaptive config: {config}")
                progress_bar.progress(0.15)
                st.write("🔍 Extracting concepts from abstracts...")
                all_concepts, all_metrics = extract_concepts_from_abstracts(df_filtered, selected_text_cols)
                st.write(f"✅ Extracted concepts from {len(all_concepts)} documents")
                progress_bar.progress(0.30)
                st.write("🧹 Filtering and normalizing concepts...")
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(all_concepts, config)
                st.write(f"✅ **{len(valid_concepts)}** valid concepts retained")
                progress_bar.progress(0.45)
                if len(valid_concepts) < 5:
                    st.error("Too few concepts extracted. Try lowering frequency thresholds.")
                    return
                st.write("🕸️ Building concept graph...")
                nx_graph = build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model, config)
                try:
                    d_prev_dict = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=4))
                except Exception:
                    d_prev_dict = {}
                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, valid_concepts, concept_to_id, config)
                st.write(f"✅ Graph: {len(valid_concepts)} nodes, {nx_graph.number_of_edges()} edges")
                progress_bar.progress(0.55)
                st.write("🧬 Generating node embeddings...")
                try:
                    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
                    node_features = torch.tensor(embeddings, dtype=torch.float32)
                except Exception:
                    node_features = torch.randn(len(valid_concepts), 384)
                st.write(f"✅ Node features: {node_features.shape}")
                progress_bar.progress(0.65)
                st.write("🤖 Training GraphSAGE...")
                def training_progress(epoch, loss):
                    progress = 0.65 + (epoch / 50) * 0.15
                    progress_bar.progress(min(1.0, progress))
                    if epoch % 10 == 0:
                        status.write(f"📊 Epoch {epoch}/50 | Loss: {loss:.4f}")
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, training_progress
                )
                st.success("✅ GNN training complete")
                progress_bar.progress(0.80)
                st.write("📈 Scoring research directions...")
                concept_properties = {}
                for concept in valid_concepts:
                    doc_indices = concept_abstract_map.get(concept, [])
                    values = []
                    for idx in doc_indices:
                        if idx < len(all_metrics):
                            for metric_values in all_metrics[idx].values():
                                values.extend(metric_values)
                    concept_properties[concept] = np.median(values) if values else 0.0
                X_feat, y_target = [], []
                for u, v in nx_graph.edges():
                    pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
                    w = nx_graph[u][v].get('weight', 1)
                    X_feat.append([pu, pv, w])
                    y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
                ridge = None
                if len(X_feat) > 5:
                    ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
                top_scores = compute_research_direction_scores(
                    gnn_model, node_features, final_emb, nx_graph, valid_concepts,
                    concept_properties, ridge, embed_model
                )
                st.write(f"✅ Scored {len(top_scores)} novel pairs")
                progress_bar.progress(0.90)
                st.write("🔬 Computing distillation metrics...")
                distill_df = compute_concept_distillation(valid_concepts, concept_abstract_map, all_texts)
                st.success("✅ Analysis complete!")
                progress_bar.progress(1.00)
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                st.session_state.analysis_data = {
                    "valid_concepts": valid_concepts,
                    "concept_to_id": concept_to_id,
                    "id_to_concept": id_to_concept,
                    "concept_abstract_map": concept_abstract_map,
                    "nx_graph": nx_graph,
                    "concept_properties": concept_properties,
                    "ridge": ridge,
                    "top_scores": top_scores,
                    "distill_df": distill_df,
                    "gnn_model": gnn_model,
                    "final_emb": final_emb,
                    "embed_model": embed_model,
                    "all_metrics": all_metrics,
                    "all_texts": all_texts,
                    "config": config,
                    "df_filtered": df_filtered  # store for timeline/growth
                }
        except Exception as e:
            st.error(f"❌ Pipeline Error: {e}")
            with st.expander("🔍 Traceback"):
                st.code(traceback.format_exc())
            return
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ─── DISPLAY RESULTS ───
    if st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        valid_concepts = data["valid_concepts"]
        concept_abstract_map = data["concept_abstract_map"]
        nx_graph = data["nx_graph"]
        top_scores = data["top_scores"]
        distill_df = data["distill_df"]
        cmap = st.session_state.get('cmap_name', 'viridis')
        top_n_graph = st.session_state.get('top_n_graph', 200)
        theme = THEME_PRESETS.get(st.session_state.get('theme', 'Bright (Default)'), THEME_PRESETS["Bright (Default)"])

        # ===== ENHANCEMENT: Extra tabs =====
        viz_tab, distill_tab, scores_tab, valid_tab, export_tab, extra_tab = st.tabs([
            "🎨 Visualization", "📊 Distillation", "🎯 Research Directions", "📐 Validation", "📥 Export", "🌟 Extra Viz"
        ])

        with viz_tab:
            st.subheader("🌐 Interactive Concept Graph")
            if nx_graph.number_of_nodes() == 0:
                st.warning("No nodes to display.")
            elif nx_graph.number_of_edges() == 0:
                st.warning("No edges — building semantic fallback")
                nx_graph = nx.complete_graph(len(valid_concepts))
                nx_graph = nx.relabel_nodes(nx_graph, {i: valid_concepts[i] for i in range(len(valid_concepts))})

            viz_choice = st.session_state.get('viz_backend', 'PyVis (Interactive)')
            physics = st.session_state.get('physics_enabled', True)
            physics_preset = st.session_state.get('effective_physics', PHYSICS_PRESETS["Stable (Default)"])

            top_n = st.session_state.get('top_n_graph', 0)

            if viz_choice == "PyVis (Interactive)":
                render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=physics,
                                   cmap_name=cmap, top_n_nodes=top_n,
                                   theme=theme, physics_preset=physics_preset)
            elif viz_choice == "Plotly 2D":
                render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=top_n,
                                       theme=theme)
            elif viz_choice == "Plotly 3D":
                render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=top_n,
                                        theme=theme)
            else:
                render_graph_fallback(nx_graph, concept_abstract_map, theme=theme)

            with st.expander("📊 Graph Metrics"):
                metrics = compute_graph_metrics(nx_graph)
                display_metric_dashboard(metrics, theme=theme)

            with st.expander("📈 Domain Hierarchy (Sunburst)"):
                # Use filters from sidebar
                cat_filter = st.session_state.get('sunburst_categories', [])
                if cat_filter:
                    st.info(f"Filtering categories: {', '.join(cat_filter)}")
                branchval = st.session_state.get('sunburst_branchvalues', 'total')
                labels, parents, values = build_category_hierarchy(valid_concepts, concept_abstract_map,
                                                                    top_n_per_category=st.session_state.get('top_n_sunburst', 0),
                                                                    categories_filter=cat_filter if cat_filter else None)
                render_sunburst_chart(labels, parents, values, cmap_name=cmap, theme=theme,
                                      branchvalues=branchval)

            with st.expander("📡 Concept Radar"):
                radar_k = st.session_state.get('top_n_radar', 15)
                if radar_k == 0:
                    radar_k = min(15, len(distill_df))
                render_radar_chart(distill_df, top_k=radar_k, cmap_name=cmap, theme=theme)

        # (Distill, Scores, Validation, Export tabs remain unchanged)
        with distill_tab:
            st.subheader("🔍 Concept Distillation Efficiency")
            top_n = st.slider("Show Top N", 10, min(200, len(distill_df)), 50, key="distill_top_n")
            display_df = distill_df.head(top_n)
            st.dataframe(display_df, use_container_width=True)
            st.markdown("**📈 Efficiency vs Frequency:**")
            chart_df = display_df.set_index('concept')[['distillation_efficiency']]
            st.bar_chart(chart_df)
            st.markdown("**📊 Multi-Metric Comparison:**")
            metric_cols = [c for c in ['frequency', 'tfidf_weight', 'semantic_density', 'coherence_score']
                           if c in display_df.columns]
            if metric_cols:
                compare_df = display_df[['concept'] + metric_cols].set_index('concept')
                st.line_chart(compare_df)

        with scores_tab:
            st.subheader("🎯 Top Research Direction Recommendations")
            if top_scores.empty:
                st.info("No novel pairs scored. The graph may be too dense or too sparse.")
            else:
                st.write(f"Top {len(top_scores)} novel concept pairs:")
                st.dataframe(top_scores[['concept_u', 'concept_v', 'composite_score',
                                         'gnn_affinity', 'semantic_novelty',
                                         'expected_property_gain', 'feasibility_score']].head(20),
                            use_container_width=True)
                csv_scores = top_scores.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Scores (CSV)", data=csv_scores,
                                  file_name="research_directions.csv", mime="text/csv")

        with valid_tab:
            st.subheader("📐 Mathematical Validation")
            val_metrics = validate_graph_metrics(nx_graph, valid_concepts)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Modularity", f"{val_metrics.get('modularity', 0):.3f}")
            col2.metric("Silhouette", f"{val_metrics.get('silhouette_score', 0):.3f}")
            col3.metric("Communities", val_metrics.get('n_communities', 0))
            col4.metric("Significant Edges", val_metrics.get('edge_significant_count', 0))
            if not top_scores.empty:
                n_boot = st.session_state.get('bootstrap_samples', 500)
                alpha = st.session_state.get('alpha_level', 0.05)
                mean_score, ci_low, ci_high = compute_bootstrap_ci(
                    top_scores['composite_score'].values, n_bootstrap=n_boot, alpha=alpha
                )
                st.success(f"🎯 Composite Score: `{mean_score:.3f}` | {int((1-alpha)*100)}% CI: `[{ci_low:.3f}, {ci_high:.3f}]`")
            X_feat, y_target = [], []
            for u, v in nx_graph.edges():
                pu, pv = data["concept_properties"].get(u, 0), data["concept_properties"].get(v, 0)
                w = nx_graph[u][v].get('weight', 1)
                X_feat.append([pu, pv, w])
                y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
            if data["ridge"] is not None and len(X_feat) > 5:
                y_pred = data["ridge"].predict(np.array(X_feat))
                st.markdown("### 🔬 Ridge Regression (Property Prediction)")
                c1, c2, c3 = st.columns(3)
                c1.metric("R²", f"{r2_score(y_target, y_pred):.3f}")
                c2.metric("MAE", f"{mean_absolute_error(y_target, y_pred):.2f}")
                c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_target, y_pred)):.2f}")

        with export_tab:
            st.subheader("📥 Export & Post-Processing")
            export_format = st.selectbox("Format:", ["GraphML", "JSON", "CSV (Edges)", "CSV (Nodes)", "PNG"])
            if st.button("📤 Generate Export"):
                result = export_graph(nx_graph, concept_abstract_map, export_format)
                if result[0]:
                    data_bytes, mime, filename = result
                    st.download_button("💾 Save File", data=data_bytes, file_name=filename, mime=mime)
            concept_list_df = pd.DataFrame({
                'concept': valid_concepts,
                'frequency': [len(concept_abstract_map.get(c, [])) for c in valid_concepts],
                'degree': [nx_graph.degree(c) for c in valid_concepts],
                'category': [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]
            })
            csv_concepts = concept_list_df.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download Concept List (CSV)", data=csv_concepts,
                              file_name="concepts.csv", mime="text/csv")

        # ===== ENHANCEMENT: Extra Visualization Tab =====
        with extra_tab:
            st.subheader("🌟 Innovative Visualizations for Core-Shell Ag-Cu Concepts")

            # Concept Timeline (if Year exists)
            df_filtered = data.get('df_filtered', pd.DataFrame())
            if not df_filtered.empty and 'Year' in df_filtered.columns:
                with st.expander("📅 Concept Timeline (Yearly Trends)", expanded=True):
                    render_timeline(df_filtered, concept_abstract_map, valid_concepts)

            # Co-occurrence Heatmap
            with st.expander("🔥 Co-occurrence Heatmap"):
                top_n_heat = st.slider("Top N concepts for heatmap", 10, 50, 30, key='heat_top')
                render_cooccurrence_heatmap(nx_graph, valid_concepts, concept_abstract_map, top_n=top_n_heat)

            # t-SNE Projection
            with st.expander("📊 t-SNE Projection"):
                render_tsne_projection(valid_concepts, data['embed_model'], concept_abstract_map, cmap)

            # Community Detection
            with st.expander("🔮 Community Detection (Modularity)"):
                render_community_detection(nx_graph, valid_concepts)

            # Concept Growth
            if not df_filtered.empty and 'Year' in df_filtered.columns:
                with st.expander("📈 Concept Growth Rate"):
                    render_concept_growth(df_filtered, concept_abstract_map, valid_concepts)

            # Bubble Chart
            with st.expander("🫧 Concept Landscape (Degree vs Frequency)"):
                render_bubble_chart(valid_concepts, concept_abstract_map, nx_graph)

if __name__ == "__main__":
    main()
