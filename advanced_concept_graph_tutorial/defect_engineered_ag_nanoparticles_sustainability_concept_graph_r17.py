#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nanomaterials-ConceptGraph v3.0: Deep Semantic Reasoning Engine
==================================================================================
Publication-ready concept graph builder with ontology-aware reasoning,
embedding-based semantic equivalence, hierarchical taxonomy, cross-domain
inference, entity disambiguation, and cause-effect relationship extraction.

v3.0 ENHANCEMENTS (Deep Semantic Reasoning Engine):
- Ontology-aware concept resolution with synonym dictionaries & hypernym/hyponym chains
- Embedding-based semantic equivalence detection (BLAS-optimized batch resolution)
- Hierarchical concept taxonomy (is-a relationships, material->microstructure->property)
- Cross-reference reasoning (Process -> Microstructure -> Property bridge inference)
- Entity disambiguation (context-window aware: "phase" in thermodynamics vs. metallurgy)
- Cause-effect relationship extraction (linguistic trigger patterns)
- N1, N2... abbreviated node labels rendered inside circle nodes with HTML legend
- Edge value inspection (weight, type, confidence, inference status on hover/tooltips)
- Symbol-based Sunburst chart (✦★●■▲◆ hierarchical symbol chains)
- Extra visualizations: t-SNE projection, keyword burst detection, semantic drift,
  co-occurrence heatmap, network motif analysis, temporal trend analysis
- Reasoning Dashboard: edge type distribution, inferred causal chains, concept type stats
- Defect engineering domain integration: vacancy, dislocation, twin, stacking fault taxonomy

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn

Run: streamlit run nanomaterials_concept_graph_v3.py

Place JSON files in ./json_metadatabase/ folder next to this script.
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
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from dataclasses import dataclass, field

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
from scipy.cluster.hierarchy import dendrogram, linkage

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
    page_title="Nanomaterials-ConceptGraph v3.0: Deep Semantic Reasoning Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PATHS & DIRECTORIES
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)

# ==========================================
# v3.0: ONTOLOGY ARCHITECTURE
# ==========================================
class ConceptType(Enum):
    """Semantic classification of nanomaterials concepts."""
    MATERIAL = "material"
    PROCESS = "process"
    PROPERTY = "property"
    MICROSTRUCTURE = "microstructure"
    PHENOMENON = "phenomenon"
    METHOD = "method"
    PARAMETER = "parameter"
    DEFECT = "defect"
    GENERAL = "general"

class RelationshipType(Enum):
    """Types of relationships between concepts."""
    CAUSES = "causes"
    RESULTS_IN = "results_in"
    INFLUENCES = "influences"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    CO_OCCURS = "co_occurs"
    SEMANTIC = "semantic"
    INFERRED = "inferred"
    BRIDGE = "bridge"
    PART_OF = "part_of"
    HAS_PROPERTY = "has_property"
    PRODUCES = "produces"

@dataclass
class ConceptNode:
    """Rich concept node with hierarchical and relational metadata."""
    canonical_name: str
    concept_type: ConceptType
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    definition: str = ""
    domain: str = "nanomaterials"

    def add_synonym(self, synonym: str):
        self.synonyms.add(synonym.lower().strip())

    def add_hypernym(self, hypernym: str):
        self.hypernyms.add(hypernym.lower().strip())

    def add_hyponym(self, hyponym: str):
        self.hyponyms.add(hyponym.lower().strip())

@dataclass
class Relationship:
    """Typed relationship between two concepts with confidence scoring."""
    source: str
    target: str
    rel_type: RelationshipType
    confidence: float = 1.0
    inferred: bool = False
    evidence: str = ""

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'rel_type': self.rel_type.value,
            'confidence': self.confidence,
            'inferred': self.inferred,
            'evidence': self.evidence
        }

# ==========================================
# COLORMAP REGISTRY
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
# ROBUST JSON LOADER
# ==========================================
def robust_load_file(filepath: Path):
    text = filepath.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"File is empty (0 bytes or only whitespace).")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    sanitized = re.sub(r'NaN', 'null', text)
    sanitized = re.sub(r'Infinity', 'null', sanitized)
    sanitized = re.sub(r'-Infinity', 'null', sanitized)
    sanitized = re.sub(r',(\s*[}\]])', r'', sanitized)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if records:
        return records
    try:
        df = pd.read_csv(filepath)
        return df.to_dict(orient="records")
    except Exception:
        pass
    preview = text[:300]
    raise ValueError(f"Could not parse {filepath.name}. First 200 chars: {preview[:200]}...")

@st.cache_data(show_spinner=False)
def load_all_json_files(directory):
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        return []
    loaded = []
    for fp in files:
        try:
            data = robust_load_file(fp)
            if isinstance(data, list):
                loaded.append((str(fp.name), data))
            elif isinstance(data, dict):
                loaded.append((str(fp.name), [data]))
            else:
                loaded.append((str(fp.name), []))
        except Exception as e:
            st.error(f"Error loading `{fp.name}`: {e}")
            try:
                raw_bytes = fp.read_bytes()[:300]
                hex_str = raw_bytes.hex()
                formatted = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
                st.code("Hex preview (first {} bytes):\n{}".format(len(raw_bytes), formatted), language="text")
            except Exception:
                pass
    return loaded

@st.cache_data(show_spinner=False)
def build_master_dataframe(file_records):
    rows = []
    for fname, records in file_records:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec["_source_file"] = fname
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    df = df.replace({float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA})
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


# ==========================================
# NANOMATERIALS DOMAIN KNOWLEDGE BASE
# ==========================================

NANOMATERIALS_PATTERNS = [
    r'\b(?:nanotwinned|nt)\s+cu(?:pper)?\b',
    r'\bcu@ag\b',
    r'\bcu/ag\b',
    r'\bcore[-\s]?shell\s+(?:cu|ag|cu@ag|cu/ag)\b',
    r'\bdefect[-\s]?engineered\s+(?:ag|silver)\b',
    r'\btwin\s+boundary\b',
    r'\bcoherent\s+twin\s+boundary\b',
    r'\bincoherent\s+twin\s+boundary\b',
    r'\bstacking\s+fault\b',
    r'\bstacking\s+fault\s+energy\b',
    r'\bdislocation\b',
    r'\bgrain\s+boundary\b',
    r'\belectrodeposition\b',
    r'\bsputtering\b',
    r'\bchemical\s+vapor\s+deposition\b',
    r'\bdensity\s+functional\s+theory\b',
    r'\bmolecular\s+dynamics\b',
    r'\byield\s+strength\b',
    r'\btensile\s+strength\b',
    r'\belastic\s+modulus\b',
    r'\bhardness\b',
    r'\bductility\b',
    r'\belongation\b',
    r'\btransmission\s+electron\s+microscopy\b',
    r'\belectron\s+backscatter\s+diffraction\b',
    r'\bx[-\s]?ray\s+diffraction\b',
    r'\bnano(?:particle|wire|rod|sheet|plate|cube|sphere|cluster|composite|structure|crystal|tube)\b',
    r'\b(?:equal[-\s]?channel\s+angular\s+pressing|ecap)\b',
    r'\bhigh[-\s]?pressure\s+torsion\b',
    r'\baccumulative\s+roll\s+bonding\b',
    r'\bfriction\s+stir\s+processing\b',
    r'\bsevere\s+plastic\s+deformation\b',
    r'\bphase\s+field\b',
    r'\bfinite\s+element\b',
    r'\b(?:machine\s+learning|neural\s+network|deep\s+learning)\b',
    r'\b(?:vacancy|interstitial|frenkel|schottky)\b',
    r'\b(?:irradiation|radiation)\s+(?:damage|defect)\b',
    r'\bdefect\s+(?:engineering|design|tailoring)\b',
]

ALL_DOMAIN_KEYWORDS = [
    "nanotwinned", "twin", "cu", "ag", "silver", "copper", "nanoparticle", "nanowire",
    "nanorod", "nanosheet", "nanoplate", "nanocube", "nanosphere", "nanocluster",
    "nanocomposite", "nanostructure", "nanocrystal", "nanotube", "thin film", "coating",
    "electrodeposition", "sputtering", "cvd", "ald", "pld", "mbe", "evaporation",
    "ecap", "hpt", "arb", "fsp", "spd", "ball milling", "cryorolling", "annealing",
    "recrystallization", "grain growth", "tem", "hrtem", "stem", "ebsd", "xrd", "apt",
    "eds", "eels", "synchrotron", "dft", "molecular dynamics", "md", "phase field",
    "finite element", "ddd", "machine learning", "neural network", "gnn", "cnn",
    "yield strength", "uts", "hardness", "elastic modulus", "young modulus", "ductility",
    "elongation", "fracture", "fatigue", "creep", "wear", "damage", "twin boundary",
    "stacking fault", "dislocation", "grain boundary", "vacancy", "interstitial",
    "defect engineering", "irradiation", "sfe", "ctb", "itb", "crss", "gnd", "ssd",
    "electrical conductivity", "resistivity", "thermal conductivity", "thermal expansion",
    "corrosion", "oxidation", "electrochemical", "catalytic", "electrocatalysis",
    "interconnect", "tsv", "solder", "conductive ink", "flexible", "transparent",
]

NANOMATERIALS_CATEGORY_MAPPING = {
    r'\bnanotwinned\b': 'nanotwinned_copper',
    r'\bnt\s+cu\b': 'nanotwinned_copper',
    r'\bcu@ag\b': 'core_shell_cuag',
    r'\bcu/ag\b': 'core_shell_cuag',
    r'\bcore[-\s]?shell\b': 'core_shell_cuag',
    r'\bdefect[-\s]?engineered\s+ag\b': 'defect_engineered_ag',
    r'\bdefect[-\s]?engineered\s+silver\b': 'defect_engineered_ag',
    r'\btwin\s+boundary\b': 'microstructure',
    r'\bstacking\s+fault\b': 'microstructure',
    r'\bdislocation\b': 'microstructure',
    r'\bgrain\s+boundary\b': 'microstructure',
    r'\bvacancy\b': 'defect',
    r'\binterstitial\b': 'defect',
    r'\belectrodeposition\b': 'synthesis',
    r'\bsputtering\b': 'synthesis',
    r'\bcvd\b': 'synthesis',
    r'\bald\b': 'synthesis',
    r'\becap\b': 'synthesis',
    r'\bhpt\b': 'synthesis',
    r'\btem\b': 'characterization',
    r'\bhrtem\b': 'characterization',
    r'\bebsd\b': 'characterization',
    r'\bxrd\b': 'characterization',
    r'\bdft\b': 'computational',
    r'\bmolecular\s+dynamics\b': 'computational',
    r'\byield\s+strength\b': 'mechanical_property',
    r'\buts\b': 'mechanical_property',
    r'\bhardness\b': 'mechanical_property',
    r'\belastic\s+modulus\b': 'mechanical_property',
    r'\bductility\b': 'mechanical_property',
    r'\bfracture\b': 'mechanical_property',
    r'\bfatigue\b': 'mechanical_property',
    r'\belectrical\s+conductivity\b': 'functional_property',
    r'\bthermal\s+conductivity\b': 'functional_property',
    r'\bcorrosion\b': 'functional_property',
    r'\bcatalytic\b': 'functional_property',
    r'\bdefect\s+engineering\b': 'defect_engineering',
    r'\birradiation\b': 'irradiation',
}

# ==========================================
# CONCEPT VALIDATION & NORMALIZATION
# ==========================================
def is_valid_nanomaterials_concept(concept: str) -> bool:
    """Validate if a concept string is meaningful for nanomaterials domain."""
    if not concept or len(concept) < 3:
        return False
    if concept.isdigit():
        return False
    # Must contain at least one alphabetic character
    if not any(c.isalpha() for c in concept):
        return False
    # Reject pure stopwords
    stopwords = {"the", "and", "for", "with", "from", "this", "that", "these", "those",
                 "are", "was", "were", "been", "have", "has", "had", "will", "would",
                 "could", "should", "may", "might", "can", "shall", "about", "into",
                 "over", "such", "than", "only", "also", "its", "our", "out", "all",
                 "use", "used", "using", "based", "via", "due", "both", "each", "more",
                 "most", "some", "any", "many", "much", "very", "well", "high", "low",
                 "new", "different", "same", "various", "several", "certain", "particular"}
    words = concept.lower().split()
    if all(w in stopwords for w in words):
        return False
    return True

def normalize_nanomaterials_term(concept: str) -> str:
    """Normalize nanomaterials terminology for consistent concept representation."""
    concept = concept.lower().strip()
    # Normalize nanoparticle variants
    concept = re.sub(r'\bnano[-\s]?particle\b', 'nanoparticle', concept)
    concept = re.sub(r'\bnano[-\s]?particles\b', 'nanoparticle', concept)
    concept = re.sub(r'\bnps\b', 'nanoparticle', concept)
    # Normalize nanotwinned copper variants
    concept = re.sub(r'\bnanotwinned\s+cu(?:pper)?\b', 'nanotwinned copper', concept)
    concept = re.sub(r'\bnt\s+cu\b', 'nanotwinned copper', concept)
    # Normalize core-shell
    concept = re.sub(r'\bcore[-\s]?shell\s+cu@ag\b', 'core_shell_cuag', concept)
    concept = re.sub(r'\bcore[-\s]?shell\s+cu/ag\b', 'core_shell_cuag', concept)
    # Normalize defect-engineered silver
    concept = re.sub(r'\bdefect[-\s]?engineered\s+ag\b', 'defect_engineered_ag', concept)
    concept = re.sub(r'\bdefect[-\s]?engineered\s+silver\b', 'defect_engineered_ag', concept)
    # Normalize twin boundary variants
    concept = re.sub(r'\bcoherent\s+twin\s+boundary\b', 'ctb', concept)
    concept = re.sub(r'\bincoherent\s+twin\s+boundary\b', 'itb', concept)
    concept = re.sub(r'\btwin\s+boundary\b', 'twin_boundary', concept)
    # Normalize nanoparticle shape variants
    concept = re.sub(r'\bnano[-\s]?wire\b', 'nanowire', concept)
    concept = re.sub(r'\bnano[-\s]?rod\b', 'nanorod', concept)
    concept = re.sub(r'\bnano[-\s]?sheet\b', 'nanosheet', concept)
    concept = re.sub(r'\bnano[-\s]?plate\b', 'nanoplate', concept)
    concept = re.sub(r'\bnano[-\s]?cube\b', 'nanocube', concept)
    concept = re.sub(r'\bnano[-\s]?sphere\b', 'nanosphere', concept)
    concept = re.sub(r'\bnano[-\s]?cluster\b', 'nanocluster', concept)
    concept = re.sub(r'\bnano[-\s]?composite\b', 'nanocomposite', concept)
    concept = re.sub(r'\bnano[-\s]?structure\b', 'nanostructure', concept)
    concept = re.sub(r'\bnano[-\s]?crystal\b', 'nanocrystal', concept)
    concept = re.sub(r'\bnano[-\s]?tube\b', 'nanotube', concept)
    # Normalize mechanical properties
    concept = re.sub(r'\byield\s*strength\b', 'yield strength', concept)
    concept = re.sub(r'\bultimate\s*tensile\s*strength\b', 'uts', concept)
    concept = re.sub(r'\btensile\s*strength\b', 'uts', concept)
    concept = re.sub(r'\belastic\s*modulus\b', 'young modulus', concept)
    concept = re.sub(r'\bstacking\s*fault\s*energy\b', 'sfe', concept)
    concept = re.sub(r'\bcoherent\s*twin\s*boundary\b', 'ctb', concept)
    concept = re.sub(r'\bincoherent\s*twin\s*boundary\b', 'itb', concept)
    concept = re.sub(r'\bcritical\s*resolved\s*shear\s*stress\b', 'crss', concept)
    concept = re.sub(r'\bgeometrically\s*necessary\s*dislocation\b', 'gnd', concept)
    concept = re.sub(r'\bstatistically\s*stored\s*dislocation\b', 'ssd', concept)
    concept = re.sub(r'\bstacking\s*fault\s*tetrahedron\b', 'sft', concept)
    # v3.0: Defect engineering normalizations
    concept = re.sub(r'\bpoint\s*defect\b', 'point_defect', concept)
    concept = re.sub(r'\blattice\s*vacancy\b', 'vacancy', concept)
    concept = re.sub(r'\bself\s*interstitial\b', 'interstitial', concept)
    concept = re.sub(r'\bfrenkel\s*pair\b', 'frenkel_pair', concept)
    concept = re.sub(r'\bschottky\s*defect\b', 'schottky_defect', concept)
    concept = re.sub(r'\bedge\s*dislocation\b', 'edge_dislocation', concept)
    concept = re.sub(r'\bscrew\s*dislocation\b', 'screw_dislocation', concept)
    concept = re.sub(r'\bshockley\s*partial\b', 'shockley_partial', concept)
    concept = re.sub(r'\bfrank\s*partial\b', 'frank_partial', concept)
    concept = re.sub(r'\bdislocation\s*loop\b', 'dislocation_loop', concept)
    concept = re.sub(r'\bdislocation\s*dipole\b', 'dislocation_dipole', concept)
    concept = re.sub(r'\btwin\s*boundary\b', 'twin_boundary', concept)
    concept = re.sub(r'\bcoherent\s*twin\s*boundary\b', 'ctb', concept)
    concept = re.sub(r'\bincoherent\s*twin\s*boundary\b', 'itb', concept)
    concept = re.sub(r'\bstacking\s*fault\b', 'stacking_fault', concept)
    concept = re.sub(r'\bintrinsic\s*stacking\s*fault\b', 'intrinsic_sf', concept)
    concept = re.sub(r'\bextrinsic\s*stacking\s*fault\b', 'extrinsic_sf', concept)
    concept = re.sub(r'\bgrain\s*boundary\b', 'grain_boundary', concept)
    concept = re.sub(r'\bhigh\s*angle\s*grain\s*boundary\b', 'high_angle_gb', concept)
    concept = re.sub(r'\blow\s*angle\s*grain\s*boundary\b', 'low_angle_gb', concept)
    concept = re.sub(r'\bdefect\s*engineering\b', 'defect_engineering', concept)
    concept = re.sub(r'\bdefect\s*design\b', 'defect_design', concept)
    concept = re.sub(r'\birradiation\s*damage\b', 'irradiation_damage', concept)
    # Normalize synthesis
    concept = re.sub(r'\belectrodeposition\b', 'electrodeposition', concept)
    concept = re.sub(r'\belectroplating\b', 'electrodeposition', concept)
    concept = re.sub(r'\bmagnetron\s*sputtering\b', 'sputtering', concept)
    concept = re.sub(r'\bchemical\s*vapor\s*deposition\b', 'cvd', concept)
    concept = re.sub(r'\batomic\s*layer\s*deposition\b', 'ald', concept)
    concept = re.sub(r'\bequal\s*channel\s*angular\s*pressing\b', 'ecap', concept)
    concept = re.sub(r'\bhigh\s*pressure\s*torsion\b', 'hpt', concept)
    concept = re.sub(r'\baccumulative\s*roll\s*bonding\b', 'arb', concept)
    concept = re.sub(r'\bfriction\s*stir\s*processing\b', 'fsp', concept)
    # Normalize characterization
    concept = re.sub(r'\btransmission\s*electron\s*microscopy\b', 'tem', concept)
    concept = re.sub(r'\bhigh\s*resolution\s*tem\b', 'hrtem', concept)
    concept = re.sub(r'\bscanning\s*transmission\s*electron\s*microscopy\b', 'stem', concept)
    concept = re.sub(r'\belectron\s*backscatter\s*diffraction\b', 'ebsd', concept)
    concept = re.sub(r'\bx-ray\s*diffraction\b', 'xrd', concept)
    concept = re.sub(r'\batom\s*probe\s*tomography\b', 'apt', concept)
    concept = re.sub(r'\benergy\s*dispersive\s*x-ray\b', 'eds', concept)
    concept = re.sub(r'\belectron\s*energy\s*loss\s*spectroscopy\b', 'eels', concept)
    # Normalize computational
    concept = re.sub(r'\bdensity\s*functional\s*theory\b', 'dft', concept)
    concept = re.sub(r'\bab\s*initio\b', 'ab initio', concept)
    concept = re.sub(r'\bfirst\s*principles\b', 'first principles', concept)
    concept = re.sub(r'\bmolecular\s*dynamics\b', 'molecular dynamics', concept)
    concept = re.sub(r'\bphase\s*field\b', 'phase field', concept)
    concept = re.sub(r'\bfinite\s*element\b', 'finite element', concept)
    concept = re.sub(r'\bdiscrete\s*dislocation\s*dynamics\b', 'ddd', concept)
    # Normalize units
    concept = re.sub(r'\bgpa\b', 'gpa', concept)
    concept = re.sub(r'\bmpa\b', 'mpa', concept)
    concept = re.sub(r'\bnm\b', 'nm', concept)
    concept = re.sub(r'\b\u00b5m\b', 'um', concept)
    return concept

# ==========================================
# EMBEDDING MODEL LOADER
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the sentence transformer embedding model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.warning(f"Could not load sentence transformer model: {e}. Using fallback.")
        # Return a minimal fallback that won't crash
        class FallbackEmbedder:
            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                # Simple hash-based embedding as fallback
                np.random.seed(42)
                return np.array([np.random.randn(384) for _ in texts])
        return FallbackEmbedder()

# ==========================================
# ADAPTIVE CONFIGURATION
# ==========================================
def get_adaptive_config(num_abstracts: int) -> Dict:
    """Generate adaptive configuration based on dataset size."""
    base_config = {
        "MIN_CONCEPT_FREQ": max(1, min(5, num_abstracts // 100)),
        "MIN_CONCEPT_LENGTH_WORDS": 2,
        "MAX_CONCEPT_LENGTH": 10,
        "TOP_N_CONCEPTS": min(1000, max(50, num_abstracts * 2)),
        "USE_SEMANTIC_CLUSTERING": num_abstracts >= 20,
        "CLUSTER_SIMILARITY": 0.72,
        "SIMILARITY_THRESHOLD": 0.85,
        "COOCCURRENCE_WEIGHT": 0.9,
        "SEMANTIC_WEIGHT": 0.1,
        "USE_INFERENCE": True,
        "USE_CAUSAL_EXTRACTION": True,
    }
    return base_config

# ==========================================
# v3.0: ONTOLOGY & REASONING CLASSES
# ==========================================
class NanomaterialsOntology:
    """v3.0: Rich ontology for nanomaterials domain with taxonomy and embeddings."""

    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self.synonym_to_canonical: Dict[str, str] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._concept_list: List[str] = []
        self._embed_model = None
        self._build_base_ontology()

    def _build_base_ontology(self):
        """Initialize base nanomaterials ontology concepts."""
        # Materials
        self._add_concept("nanotwinned copper", ConceptType.MATERIAL,
                         synonyms={"nt cu", "nanotwinned cu", "nt copper"},
                         definition="Copper with nanoscale twin boundaries")
        self._add_concept("core_shell_cuag", ConceptType.MATERIAL,
                         synonyms={"cu@ag", "cu/ag", "core-shell cu-ag"},
                         definition="Core-shell copper-silver nanoparticles")
        self._add_concept("defect_engineered_ag", ConceptType.MATERIAL,
                         synonyms={"defect engineered silver", "ag defect"},
                         definition="Silver with engineered defect structures")

        # Microstructures
        self._add_concept("twin_boundary", ConceptType.MICROSTRUCTURE,
                         synonyms={"tb", "twin interface"},
                         hypernyms={"grain_boundary"})
        self._add_concept("ctb", ConceptType.MICROSTRUCTURE,
                         synonyms={"coherent twin boundary"},
                         hypernyms={"twin_boundary"})
        self._add_concept("itb", ConceptType.MICROSTRUCTURE,
                         synonyms={"incoherent twin boundary"},
                         hypernyms={"twin_boundary"})
        self._add_concept("stacking_fault", ConceptType.MICROSTRUCTURE,
                         synonyms={"sf", "stacking fault"},
                         hypernyms={"planar_defect"})
        self._add_concept("dislocation", ConceptType.MICROSTRUCTURE,
                         synonyms={"disl", "line defect"},
                         hypernyms={"crystal_defect"})
        self._add_concept("grain_boundary", ConceptType.MICROSTRUCTURE,
                         synonyms={"gb", "grain interface"},
                         hypernyms={"interface"})
        self._add_concept("vacancy", ConceptType.DEFECT,
                         synonyms={"lattice vacancy", "point defect"},
                         hypernyms={"point_defect"})
        self._add_concept("interstitial", ConceptType.DEFECT,
                         synonyms={"self interstitial"},
                         hypernyms={"point_defect"})

        # Properties
        self._add_concept("yield strength", ConceptType.PROPERTY,
                         synonyms={"ys", "yield stress"})
        self._add_concept("uts", ConceptType.PROPERTY,
                         synonyms={"ultimate tensile strength", "tensile strength"})
        self._add_concept("young modulus", ConceptType.PROPERTY,
                         synonyms={"elastic modulus", "e-modulus", "modulus of elasticity"})
        self._add_concept("hardness", ConceptType.PROPERTY,
                         synonyms={"hv", "vickers hardness"})
        self._add_concept("ductility", ConceptType.PROPERTY,
                         synonyms={"elongation", "strain to failure"})
        self._add_concept("sfe", ConceptType.PROPERTY,
                         synonyms={"stacking fault energy", "gsfe"})

        # Processes
        self._add_concept("electrodeposition", ConceptType.PROCESS,
                         synonyms={"electroplating", "electrochemical deposition"})
        self._add_concept("sputtering", ConceptType.PROCESS,
                         synonyms={"magnetron sputtering", "pvd"})
        self._add_concept("cvd", ConceptType.PROCESS,
                         synonyms={"chemical vapor deposition"})
        self._add_concept("ecap", ConceptType.PROCESS,
                         synonyms={"equal channel angular pressing", "ecap"})
        self._add_concept("hpt", ConceptType.PROCESS,
                         synonyms={"high pressure torsion"})
        self._add_concept("annealing", ConceptType.PROCESS,
                         synonyms={"heat treatment", "thermal treatment"})

        # Methods
        self._add_concept("tem", ConceptType.METHOD,
                         synonyms={"transmission electron microscopy", "electron microscopy"})
        self._add_concept("dft", ConceptType.METHOD,
                         synonyms={"density functional theory", "ab initio", "first principles"})
        self._add_concept("molecular dynamics", ConceptType.METHOD,
                         synonyms={"md", "atomistic simulation"})

        # Defect engineering
        self._add_concept("defect_engineering", ConceptType.PROCESS,
                         synonyms={"defect design", "defect tailoring"},
                         definition="Controlled introduction and manipulation of defects")
        self._add_concept("irradiation_damage", ConceptType.PHENOMENON,
                         synonyms={"radiation damage", "ion implantation damage"})

    def _add_concept(self, name: str, concept_type: ConceptType, synonyms: Set[str] = None,
                     hypernyms: Set[str] = None, hyponyms: Set[str] = None,
                     definition: str = ""):
        """Add a concept to the ontology."""
        node = ConceptNode(
            canonical_name=name,
            concept_type=concept_type,
            synonyms=synonyms or set(),
            hypernyms=hypernyms or set(),
            hyponyms=hyponyms or set(),
            definition=definition
        )
        self.concepts[name] = node
        # Register synonyms
        for syn in (synonyms or set()):
            self.synonym_to_canonical[syn.lower()] = name
        self.synonym_to_canonical[name.lower()] = name

    def build_embeddings(self, embed_model):
        """Build embedding cache for all ontology concepts."""
        self._embed_model = embed_model
        self._concept_list = list(self.concepts.keys())
        if self._concept_list:
            self._embeddings = embed_model.encode(self._concept_list, show_progress_bar=False, batch_size=64)

    def get_concept_type(self, concept: str) -> ConceptType:
        """Get the semantic type of a concept."""
        canonical = self.synonym_to_canonical.get(concept.lower(), concept.lower())
        if canonical in self.concepts:
            return self.concepts[canonical].concept_type
        # Heuristic fallback
        c = concept.lower()
        if any(x in c for x in ['strength', 'hardness', 'modulus', 'ductility', 'conductivity']):
            return ConceptType.PROPERTY
        elif any(x in c for x in ['electrodeposition', 'sputtering', 'annealing', 'ecap', 'hpt']):
            return ConceptType.PROCESS
        elif any(x in c for x in ['tem', 'ebsd', 'xrd', 'apt']):
            return ConceptType.METHOD
        elif any(x in c for x in ['dislocation', 'twin', 'grain', 'stacking fault', 'vacancy']):
            return ConceptType.MICROSTRUCTURE
        elif any(x in c for x in ['nanoparticle', 'nanowire', 'thin film', 'cu', 'ag']):
            return ConceptType.MATERIAL
        elif any(x in c for x in ['defect engineering', 'irradiation']):
            return ConceptType.DEFECT if 'defect' in c else ConceptType.PHENOMENON
        return ConceptType.GENERAL

    def get_hypernyms(self, concept: str) -> Set[str]:
        """Get hypernyms (is-a parents) of a concept."""
        canonical = self.synonym_to_canonical.get(concept.lower(), concept.lower())
        if canonical in self.concepts:
            return self.concepts[canonical].hypernyms
        return set()

    def infer_path(self, source: str, target: str, max_depth: int = 2) -> List[List[str]]:
        """Infer reasoning paths between concepts via shared ontology."""
        paths = []
        source_type = self.get_concept_type(source)
        target_type = self.get_concept_type(target)

        # Process -> Microstructure -> Property bridge
        if source_type == ConceptType.PROCESS and target_type == ConceptType.PROPERTY:
            # Find intermediate microstructure concepts
            for concept_name, node in self.concepts.items():
                if node.concept_type in (ConceptType.MICROSTRUCTURE, ConceptType.DEFECT):
                    paths.append([source, concept_name, target])

        # Material -> Property via Microstructure
        if source_type == ConceptType.MATERIAL and target_type == ConceptType.PROPERTY:
            for concept_name, node in self.concepts.items():
                if node.concept_type == ConceptType.MICROSTRUCTURE:
                    paths.append([source, concept_name, target])

        return paths[:10]  # Limit paths


class AdvancedConceptResolver:
    """v3.0: Resolves raw text phrases to canonical ontology concepts with disambiguation."""

    def __init__(self, ontology: NanomaterialsOntology, embed_model,
                 similarity_threshold: float = 0.85):
        self.ontology = ontology
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self._resolution_cache: Dict[str, str] = {}

    def resolve(self, phrase: str, context: str = "") -> str:
        """Resolve a raw phrase to canonical concept form."""
        phrase_lower = phrase.lower().strip()

        # Check cache
        if phrase_lower in self._resolution_cache:
            return self._resolution_cache[phrase_lower]

        # Check direct synonym match
        if phrase_lower in self.ontology.synonym_to_canonical:
            canonical = self.ontology.synonym_to_canonical[phrase_lower]
            self._resolution_cache[phrase_lower] = canonical
            return canonical

        # Check fuzzy match against ontology
        if self.ontology._embeddings is not None and self.ontology._concept_list:
            try:
                phrase_emb = self.embed_model.encode([phrase_lower], show_progress_bar=False)
                sims = cosine_similarity(phrase_emb, self.ontology._embeddings)[0]
                best_idx = np.argmax(sims)
                if sims[best_idx] >= self.similarity_threshold:
                    canonical = self.ontology._concept_list[best_idx]
                    self._resolution_cache[phrase_lower] = canonical
                    return canonical
            except Exception:
                pass

        # Context disambiguation for ambiguous terms
        if context:
            disambiguated = self._disambiguate(phrase_lower, context)
            if disambiguated:
                self._resolution_cache[phrase_lower] = disambiguated
                return disambiguated

        # Return normalized form
        normalized = normalize_nanomaterials_term(phrase_lower)
        self._resolution_cache[phrase_lower] = normalized
        return normalized

    def _disambiguate(self, phrase: str, context: str) -> Optional[str]:
        """Context-aware disambiguation for polysemous terms."""
        context_lower = context.lower()

        # "phase" disambiguation
        if phrase == "phase":
            if any(w in context_lower for w in ['thermodynamic', 'free energy', 'gibbs', 'enthalpy']):
                return "thermodynamic_phase"
            elif any(w in context_lower for w in ['microstructure', 'grain', 'precipitate', 'martensite']):
                return "microstructural_phase"
            return None

        # "twin" disambiguation
        if phrase in ["twin", "twinning"]:
            if any(w in context_lower for w in ['deformation', 'mechanical', 'stress', 'strain']):
                return "deformation_twinning"
            elif any(w in context_lower for w in ['growth', 'annealing', 'thermal']):
                return "annealing_twin"
            return "twin_boundary"

        # "defect" disambiguation
        if phrase == "defect":
            if any(w in context_lower for w in ['point defect', 'vacancy', 'interstitial']):
                return "point_defect"
            elif any(w in context_lower for w in ['dislocation', 'line defect']):
                return "dislocation"
            elif any(w in context_lower for w in ['grain boundary', 'interface']):
                return "grain_boundary"
            return None

        return None


class EnhancedConceptExtractor:
    """v3.0: Extracts concepts and relationships from text with ontology awareness."""

    # Causal trigger patterns for relationship extraction
    CAUSAL_TRIGGERS = [
        (r'\b(caused? by|results? in|leads? to|gives? rise to)\b', RelationshipType.CAUSES),
        (r'\b(results? in|produces?|generates?|yields?)\b', RelationshipType.RESULTS_IN),
        (r'\b(influences?|affects?|impacts?|modulates?)\b', RelationshipType.INFLUENCES),
        (r'\b(enhances?|improves?|increases?|boosts?)\b', RelationshipType.INFLUENCES),
        (r'\b(reduces?|decreases?|suppresses?|inhibits?)\b', RelationshipType.INFLUENCES),
    ]

    def __init__(self, ontology: NanomaterialsOntology, resolver: AdvancedConceptResolver):
        self.ontology = ontology
        self.resolver = resolver
        self._global_phrases: Dict[str, int] = defaultdict(int)
        self._global_resolutions: Dict[str, str] = {}

    def extract_from_text(self, text: str) -> Set[str]:
        """Extract concepts from text with ontology-aware resolution."""
        concepts = set()
        text_lower = text.lower()

        # Pattern-based extraction
        for pattern in NANOMATERIALS_PATTERNS:
            matches = re.findall(pattern, text, re.I)
            for m in matches:
                if isinstance(m, tuple):
                    m = m[0] if m[0] else m[1] if len(m) > 1 else str(m)
                concept = str(m).lower().strip().rstrip('.').rstrip(',')
                if len(concept) > 3 and is_valid_nanomaterials_concept(concept):
                    resolved = self.resolver.resolve(concept, context=text)
                    concepts.add(resolved)
                    self._global_phrases[concept] += 1

        # Keyword-based extraction with context
        for keyword in ALL_DOMAIN_KEYWORDS:
            for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text_lower[start:end]
                # Extract noun phrases around keyword
                context_phrases = re.findall(
                    r'\b([a-z]+(?:\s+[a-z]+){1,3})\s+(?:of|for|in|with|using|via|through|by|to|and|or)\s+' + re.escape(keyword) + r'\b',
                    context
                )
                for phrase in context_phrases:
                    concept = f"{phrase.strip()} {keyword}"
                    if is_valid_nanomaterials_concept(concept):
                        resolved = self.resolver.resolve(concept, context=text)
                        concepts.add(resolved)
                        self._global_phrases[concept] += 1

        # Direct keyword inclusion
        for keyword in ALL_DOMAIN_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                resolved = self.resolver.resolve(keyword, context=text)
                concepts.add(resolved)
                self._global_phrases[keyword] += 1

        return concepts

    def extract_relationships(self, text: str) -> List[Relationship]:
        """Extract cause-effect relationships from text."""
        relationships = []
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Extract concepts in this sentence
            concepts_in_sent = list(self.extract_from_text(sentence))
            if len(concepts_in_sent) < 2:
                continue

            # Look for causal triggers
            for trigger_pattern, rel_type in self.CAUSAL_TRIGGERS:
                if re.search(trigger_pattern, sentence, re.I):
                    # Link concepts before and after trigger
                    trigger_match = re.search(trigger_pattern, sentence, re.I)
                    if trigger_match:
                        before = sentence[:trigger_match.start()]
                        after = sentence[trigger_match.end():]

                        before_concepts = [c for c in concepts_in_sent if c.lower() in before.lower()]
                        after_concepts = [c for c in concepts_in_sent if c.lower() in after.lower()]

                        for src in before_concepts[:2]:  # Limit connections
                            for tgt in after_concepts[:2]:
                                if src != tgt:
                                    relationships.append(Relationship(
                                        source=src,
                                        target=tgt,
                                        rel_type=rel_type,
                                        confidence=0.7,
                                        evidence=sentence[:150]
                                    ))

        return relationships

    def finalize_global_resolution(self) -> Dict[str, str]:
        """Finalize global phrase-to-concept resolution map."""
        # For phrases that appear frequently, ensure consistent mapping
        for phrase, count in self._global_phrases.items():
            if phrase not in self._global_resolutions:
                self._global_resolutions[phrase] = self.resolver.resolve(phrase)
        return self._global_resolutions

def extract_concepts_from_text(text: str) -> List[str]:
    """Legacy single-text extractor (kept for backward compatibility)."""
    concepts = set()
    text_lower = text.lower()
    for pattern in NANOMATERIALS_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    noun_pattern = r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,4}(?:nanoparticle|nanowire|nanorod|nanostructure|nanocrystal|nanotube|nanosheet|nanoplate|nanocube|nanosphere|nanocluster|nanocomposite|thin\s*film|coating|layer|interface|boundary|defect|dislocation|twin|precipitate|grain|phase|structure|morphology|property|performance|mechanism|process|method|technique|analysis|simulation|model|design|optimization)\b'
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_nanomaterials_concept(concept):
            concepts.add(concept)
    for keyword in ALL_DOMAIN_KEYWORDS:
        for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text_lower[start:end]
            context_phrases = re.findall(r'\b([a-z]+(?:\s+[a-z]+){1,3})\s+(?:of|for|in|with|using|via|through|by|to|and|or)\s+' + re.escape(keyword) + r'\b', context)
            for phrase in context_phrases:
                concept = f"{phrase.strip()} {keyword}"
                if is_valid_nanomaterials_concept(concept):
                    concepts.add(concept)
    material_prop_pattern = r'\b([A-Z][a-z]+(?:\d+(?:\.\d+)?)?(?:[\s\-][A-Z][a-z]?\d*)+)\b\s+(?:with|having|exhibiting|showing|demonstrating|achieving|reaching|delivering|providing|offering)\s+(?:a\s+)?([\d\.]+\s*(?:gpa|mpa|hv|nm|um|µm|angstrom|å|wh/kg|mah/g))\b'
    matches = re.findall(material_prop_pattern, text, re.I)
    for material, value in matches:
        concept = f"{material.lower()} {value.lower()}"
        if is_valid_nanomaterials_concept(concept):
            concepts.add(concept)
    return list(concepts)

def extract_concepts_from_abstracts(df: pd.DataFrame, text_columns: List[str]) -> Tuple[List[List[str]], List[Dict]]:
    all_concepts = []
    all_metrics = []
    for idx, row in df.iterrows():
        combined_text = ""
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                combined_text += " " + str(row[col])
        metrics = {}
        strength_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:gpa|mpa)', combined_text, re.I)
        if strength_matches: metrics['strength_mpa_gpa'] = [float(m) for m in strength_matches]
        size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:nm|um|µm)', combined_text, re.I)
        if size_matches: metrics['size_nm_um'] = [float(m) for m in size_matches]
        twin_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:twin|twinning)', combined_text, re.I)
        if twin_matches: metrics['twin_fraction_pct'] = [float(m) for m in twin_matches]
        sfe_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mj/m\^2|mj/m2|erg/cm\^2)', combined_text, re.I)
        if sfe_matches: metrics['sfe_mj_m2'] = [float(m) for m in sfe_matches]
        disl_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:x\s*10\^\d+|e\+?\d+)?\s*(?:m\^-2|/m\^2|per\s*m\^2)', combined_text, re.I)
        if disl_matches: metrics['dislocation_density'] = [float(m) for m in disl_matches]
        grain_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:nm|um|µm)\s*(?:grain|grain\s*size)', combined_text, re.I)
        if grain_matches: metrics['grain_size'] = [float(m) for m in grain_matches]
        # v3.0: Defect engineering metrics
        vacancy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:x\s*10\^\d+)?\s*(?:vacancy|vacancies)', combined_text, re.I)
        if vacancy_matches: metrics['vacancy_concentration'] = [float(m) for m in vacancy_matches]
        defect_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:defects|defect)', combined_text, re.I)
        if defect_matches: metrics['defect_density'] = [float(m) for m in defect_matches]
        all_metrics.append(metrics)
        concepts = extract_concepts_from_text(combined_text)
        normalized = [normalize_nanomaterials_term(c) for c in concepts]
        all_concepts.append(normalized)
    return all_concepts, all_metrics

def extract_concepts_parallel(df: pd.DataFrame, text_columns: List[str], 
                               extractor: EnhancedConceptExtractor,
                               max_workers: int = 8) -> Tuple[List[List[str]], List[Dict], List[Relationship]]:
    """
    v3.0: Parallel extraction with deferred global resolution and relationship extraction.
    Returns raw concepts, metrics, and extracted relationships.
    """
    all_concepts = [[] for _ in range(len(df))]
    all_metrics = [{} for _ in range(len(df))]
    all_relationships = []

    def process_row(idx_row):
        idx, row = idx_row
        combined_text = ""
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                combined_text += " " + str(row[col])

        metrics = {}
        strength_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:gpa|mpa)', combined_text, re.I)
        if strength_matches: metrics['strength_mpa_gpa'] = [float(m) for m in strength_matches]
        size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:nm|um|µm)', combined_text, re.I)
        if size_matches: metrics['size_nm_um'] = [float(m) for m in size_matches]

        # v3.0: Extract concepts and relationships
        concepts = extractor.extract_from_text(combined_text)
        relationships = extractor.extract_relationships(combined_text)

        return idx, list(concepts), metrics, relationships

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, (i, row)): i for i, row in df.iterrows()}
        for future in as_completed(futures):
            idx, concepts, metrics, relationships = future.result()
            all_concepts[idx] = concepts
            all_metrics[idx] = metrics
            all_relationships.extend(relationships)

    return all_concepts, all_metrics, all_relationships

def cluster_similar_concepts(valid_concepts: List[str], embed_model, similarity_threshold: float = 0.75):
    if len(valid_concepts) < 5:
        return valid_concepts, {c: c for c in valid_concepts}
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1 - similarity_threshold,
            linkage='average', metric='cosine'
        ).fit(embeddings)
        cluster_members = defaultdict(list)
        concept_to_cluster = {}
        for idx, label in enumerate(clustering.labels_):
            concept = valid_concepts[idx]
            cluster_members[label].append(concept)
            concept_to_cluster[concept] = label
        cluster_representatives = {}
        for label, members in cluster_members.items():
            def score(m):
                domain_hits = sum(1 for kw in ALL_DOMAIN_KEYWORDS if kw.lower() in m.lower())
                return (domain_hits, -len(m))
            representative = max(members, key=score)
            cluster_representatives[label] = representative
        final_mapping = {c: cluster_representatives[label] for c, label in concept_to_cluster.items()}
        return list(cluster_representatives.values()), final_mapping
    except Exception as e:
        return valid_concepts, {c: c for c in valid_concepts}

def normalize_and_filter_concepts(all_concepts: List[List[str]], config: Dict, 
                                   ontology: NanomaterialsOntology = None) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc and is_valid_nanomaterials_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
    min_freq = config.get("MIN_CONCEPT_FREQ", 5)
    min_words = config.get("MIN_CONCEPT_LENGTH_WORDS", 2)
    max_words = config.get("MAX_CONCEPT_LENGTH", 10)
    valid_concepts = [c for c, cnt in concept_counts.items()
                      if cnt >= min_freq and min_words <= len(c.split()) <= max_words]
    if config.get("USE_SEMANTIC_CLUSTERING", False) and len(valid_concepts) > 50:
        try:
            embed_model = load_embedding_model()
            valid_concepts, concept_to_cluster = cluster_similar_concepts(
                valid_concepts, embed_model,
                similarity_threshold=config.get("CLUSTER_SIMILARITY", 0.72)
            )
            new_abstract_map = defaultdict(list)
            for orig_concept, docs in concept_abstract_map.items():
                clustered = concept_to_cluster.get(orig_concept, orig_concept)
                if clustered in valid_concepts:
                    new_abstract_map[clustered].extend(docs)
            concept_abstract_map = new_abstract_map
        except Exception as e:
            st.warning(f"Semantic clustering skipped: {e}")
    valid_concepts = sorted(valid_concepts, key=lambda c: concept_counts[c], reverse=True)
    top_n = config.get("TOP_N_CONCEPTS", 1000)
    if len(valid_concepts) > top_n:
        valid_concepts = valid_concepts[:top_n]
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

def abstract_concepts_to_categories(concepts: List[str]) -> Dict[str, str]:
    concept_to_abstract = {}
    for concept in concepts:
        matched = False
        for pattern, category in NANOMATERIALS_CATEGORY_MAPPING.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            if any(re.search(p, concept, re.I) for p in [r'\bnanotwinned', r'\bnt\s*cu', r'\btwin']):
                concept_to_abstract[concept] = 'nanotwinned_copper'
            elif any(re.search(p, concept, re.I) for p in [r'\bcu@ag', r'\bcu/ag', r'\bcore\s*shell']):
                concept_to_abstract[concept] = 'core_shell_cuag'
            elif any(re.search(p, concept, re.I) for p in [r'\bdefect\s*engineered', r'\bag\s*defect', r'\bsilver\s*defect']):
                concept_to_abstract[concept] = 'defect_engineered_ag'
            # v3.0: Defect engineering categories
            elif any(re.search(p, concept, re.I) for p in [r'\bdefect\s*engineering', r'\bdefect\s*design', r'\bdefect\s*tailoring']):
                concept_to_abstract[concept] = 'defect_engineering'
            elif any(re.search(p, concept, re.I) for p in [r'\birradiation', r'\bradiation\s*defect']):
                concept_to_abstract[concept] = 'irradiation_damage'
            elif any(re.search(p, concept, re.I) for p in [r'\bvacancy', r'\binterstitial', r'\bfrenkel', r'\bschottky']):
                concept_to_abstract[concept] = 'point_defect'
            elif any(re.search(p, concept, re.I) for p in [r'\bdislocation\s*loop', r'\bdislocation\s*dipole']):
                concept_to_abstract[concept] = 'dislocation_substructure'
            elif any(re.search(p, concept, re.I) for p in [r'\bvoid', r'\bcavity', r'\bpore']):
                concept_to_abstract[concept] = 'volume_defect'
            elif any(re.search(p, concept, re.I) for p in [r'\bprecipitate', r'\bgp\s*zone', r'\bguinier']):
                concept_to_abstract[concept] = 'precipitation'
            else:
                concept_to_abstract[concept] = 'general'
    return concept_to_abstract

# ==========================================
# CONCEPT DISTILLATION
# ==========================================
def compute_concept_distillation(valid_concepts: List[str], concept_abstract_map: Dict[str, List[int]],
                                  all_texts: List[str]) -> pd.DataFrame:
    distill_data = []
    doc_corpus = []
    for c in valid_concepts:
        doc_text = " ".join([all_texts[i] for i in concept_abstract_map.get(c, []) if i < len(all_texts)])
        doc_corpus.append(doc_text)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english', max_features=5000)
    try:
        tfidf_matrix = tfidf.fit_transform(doc_corpus)
        tfidf_scores = tfidf_matrix.max(axis=1).A1
    except Exception:
        tfidf_scores = np.ones(len(valid_concepts))
    embed_model = load_embedding_model()
    for i, c in enumerate(valid_concepts):
        freq = len(concept_abstract_map.get(c, []))
        semantic_density = float(tfidf_scores[i])
        coherence = 0.0
        if freq > 1 and doc_corpus[i].strip():
            try:
                words = doc_corpus[i].split()[:50]
                concept_embeddings = embed_model.encode(words, show_progress_bar=False, batch_size=32)
                if len(concept_embeddings) > 1:
                    sim_matrix = cosine_similarity(concept_embeddings)
                    coherence = float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))
            except Exception:
                coherence = 0.0
        distill_data.append({
            "concept": c, "frequency": freq, "tfidf_weight": semantic_density,
            "semantic_density": semantic_density, "coherence_score": float(coherence),
            "distillation_efficiency": float(semantic_density * np.log1p(freq) * (0.5 + 0.5 * coherence))
        })
    return pd.DataFrame(distill_data).sort_values("distillation_efficiency", ascending=False)

# ==========================================
# v3.0: REASONING-ENHANCED GRAPH BUILDER
# ==========================================
class ReasoningEnhancedGraphBuilder:
    """
    v3.0: Multi-layered graph construction with ontology reasoning,
    hierarchical edges, cross-domain bridge inference, and cause-effect relationships.
    """
    def __init__(self, ontology: NanomaterialsOntology, extractor: EnhancedConceptExtractor):
        self.ontology = ontology
        self.extractor = extractor
        self.reasoning_paths = []
        self.inferred_edges = set()
        self.extracted_relationships = []

    def build_graph(self, all_concepts, valid_concepts, concept_to_id, 
                    relationships: List[Relationship] = None,
                    embed_model=None, config: Dict = None) -> nx.Graph:
        if config is None:
            config = get_adaptive_config(3000)

        nx_graph = nx.Graph()

        # 1. Add nodes with enriched attributes (v3.0: concept type from ontology)
        for c in valid_concepts:
            concept_type = self.ontology.get_concept_type(c)
            nx_graph.add_node(c, frequency=0, concept_type=concept_type.value,
                            hierarchy_path=self.ontology.get_hypernyms(c))

        # 2. Layer 1: Co-occurrence (Observed)
        for concepts in all_concepts:
            valid_in_doc = [c for c in concepts if c in concept_to_id]
            for i in range(len(valid_in_doc)):
                for j in range(i + 1, len(valid_in_doc)):
                    u, v = valid_in_doc[i], valid_in_doc[j]
                    if nx_graph.has_edge(u, v):
                        nx_graph[u][v]['weight'] += 1
                        nx_graph[u][v]['cooccurrence'] += 1
                    else:
                        nx_graph.add_edge(u, v, weight=1, cooccurrence=1, semantic=0, 
                                          edge_type='cooccurrence', inferred=False,
                                          confidence=1.0, evidence="")
                    nx_graph.nodes[u]['frequency'] = nx_graph.nodes[u].get('frequency', 0) + 1
                    nx_graph.nodes[v]['frequency'] = nx_graph.nodes[v].get('frequency', 0) + 1

        # 3. Layer 2: Semantic (Embeddings) - with v3.0 enrichment
        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(nx_graph, valid_concepts, embed_model, config)

        # 4. Layer 3: Hierarchical (Taxonomy) - v3.0: Ontology-based is-a relationships
        self._add_hierarchical_edges(nx_graph, valid_concepts)

        # 5. Layer 4: Cause-Effect (Extracted from text) - v3.0
        if relationships and config.get('USE_CAUSAL_EXTRACTION', True):
            self._add_causal_edges(nx_graph, relationships)

        # 6. Layer 5: Cross-Domain Bridges (Inferred) - v3.0
        if config.get('USE_INFERENCE', True):
            self._infer_cross_domain_bridges(nx_graph, valid_concepts)

        # 7. Final weight computation
        cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.9)
        sem_weight = config.get("SEMANTIC_WEIGHT", 0.1)
        for u, v, data in nx_graph.edges(data=True):
            cooc = data.get('cooccurrence', 0)
            sem = data.get('semantic', 0)
            data['weight'] = cooc_weight * cooc + sem_weight * sem

        return nx_graph

    def _add_semantic_edges(self, nx_graph, valid_concepts, embed_model, config):
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        sim_matrix = cosine_similarity(embeddings)
        sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
        for i, c1 in enumerate(valid_concepts):
            for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                if c1 == c2 or nx_graph.has_edge(c1, c2):
                    continue
                sim = sim_matrix[i][j]
                if sim > sim_thresh and (nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3):
                    nx_graph.add_edge(c1, c2, weight=sim * 2, cooccurrence=0,
                                     semantic=sim, edge_type='semantic', inferred=False,
                                     confidence=sim, evidence="Embedding similarity")

    def _add_hierarchical_edges(self, nx_graph, valid_concepts):
        """v3.0: Add hypernym/hyponym edges based on ontology taxonomy."""
        for concept in valid_concepts:
            if concept in self.ontology.concepts:
                node = self.ontology.concepts[concept]
                for hyp in node.hypernyms:
                    if hyp in valid_concepts and not nx_graph.has_edge(concept, hyp):
                        nx_graph.add_edge(concept, hyp, weight=1.0, edge_type='hypernym', 
                                         inferred=True, confidence=0.9, 
                                         evidence=f"Ontology: {concept} is-a {hyp}")
                for hyponym in node.hyponyms:
                    if hyponym in valid_concepts and not nx_graph.has_edge(concept, hyponym):
                        nx_graph.add_edge(concept, hyponym, weight=1.0, edge_type='hyponym',
                                         inferred=True, confidence=0.9,
                                         evidence=f"Ontology: {hyponym} is-a {concept}")

    def _add_causal_edges(self, nx_graph, relationships: List[Relationship]):
        """v3.0: Add cause-effect edges extracted from text."""
        for rel in relationships:
            if rel.source in nx_graph.nodes() and rel.target in nx_graph.nodes():
                if not nx_graph.has_edge(rel.source, rel.target):
                    nx_graph.add_edge(rel.source, rel.target, 
                                     weight=rel.confidence * 2,
                                     edge_type=rel.rel_type.value,
                                     inferred=rel.inferred,
                                     confidence=rel.confidence,
                                     evidence=rel.evidence[:200])
                else:
                    # Strengthen existing edge with causal evidence
                    nx_graph[rel.source][rel.target]['causal_confidence'] = rel.confidence
                    nx_graph[rel.source][rel.target]['causal_evidence'] = rel.evidence[:200]

    def _infer_cross_domain_bridges(self, nx_graph, valid_concepts):
        """v3.0: Infer edges bridging Process -> Property through shared Microstructure/Defect."""
        process_nodes = [c for c in valid_concepts 
                        if self.ontology.get_concept_type(c) == ConceptType.PROCESS]
        property_nodes = [c for c in valid_concepts 
                         if self.ontology.get_concept_type(c) == ConceptType.PROPERTY]

        for proc in process_nodes:
            for prop in property_nodes:
                if not nx_graph.has_edge(proc, prop):
                    # Check ontology for a valid path
                    paths = self.ontology.infer_path(proc, prop, max_depth=2)
                    if paths:
                        nx_graph.add_edge(proc, prop, weight=0.8, semantic=0.8, 
                                         edge_type='bridge', inferred=True, 
                                         confidence=0.75, 
                                         path=" -> ".join(paths[0]),
                                         evidence=f"Inferred via: {' -> '.join(paths[0])}")
                        self.inferred_edges.add((proc, prop))
                        self.reasoning_paths.append(paths[0])

# ==========================================
# GRAPH CONSTRUCTION (Legacy wrapper for compatibility)
# ==========================================
def build_hybrid_graph(all_concepts: List[List[str]], valid_concepts: List[str],
                        concept_to_id: Dict[str, int], embed_model=None, config: Dict = None) -> nx.Graph:
    """Legacy wrapper - now uses ReasoningEnhancedGraphBuilder internally."""
    if config is None:
        config = get_adaptive_config(3000)

    # Create minimal ontology for legacy compatibility
    ontology = NanomaterialsOntology()
    if embed_model:
        ontology.build_embeddings(embed_model)
    resolver = AdvancedConceptResolver(ontology, embed_model or load_embedding_model())
    extractor = EnhancedConceptExtractor(ontology, resolver)
    builder = ReasoningEnhancedGraphBuilder(ontology, extractor)

    return builder.build_graph(all_concepts, valid_concepts, concept_to_id, 
                               relationships=[], embed_model=embed_model, config=config)

def sample_edges_for_training(nx_graph: nx.Graph, valid_concepts: List[str],
                               concept_to_id: Dict[str, int], config: Dict = None) -> Tuple[List[Tuple], List[Tuple]]:
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs
    target_negs = min(len(pos_pairs) * 3 if pos_pairs else 30, 5000)
    attempts = 0
    max_attempts = 50000
    try:
        path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=3))
    except Exception:
        path_lengths = {}
    while len(neg_pairs) < target_negs and attempts < max_attempts:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            attempts += 1
            continue
        dist = path_lengths.get(u_c, {}).get(v_c, 999)
        if dist == 2 or dist == 3:
            neg_pairs.append((u_idx, v_idx))
        elif dist == 999 and np.random.rand() < 0.1:
            neg_pairs.append((u_idx, v_idx))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            neg_pairs.append((u_idx, v_idx))
    return pos_pairs, neg_pairs

# ==========================================
# GNN MODEL (Unchanged from v2.0)
# ==========================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
    def forward(self, adj_indices, adj_values, num_nodes, h, pos_u, pos_v, neg_u, neg_v):
        A = sparse.FloatTensor(adj_indices, adj_values, torch.Size([num_nodes, num_nodes])).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        return pos_scores, neg_scores, h2

def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs,
              progress_callback=None, epochs: int = 50, lr: float = 1e-3):
    num_nodes = len(concept_to_id)
    in_dim = node_features.shape[1] if node_features.numel() > 0 else 384
    if not pos_pairs:
        nodes = list(concept_to_id.values())
        if len(nodes) >= 2:
            pos_pairs = [(nodes[0], nodes[1])]
        else:
            raise ValueError("Cannot train GNN with fewer than 2 concepts")
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    target_device = node_features.device if node_features.numel() > 0 else torch.device('cpu')
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=target_device)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=target_device)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else torch.tensor([], dtype=torch.long, device=target_device)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else torch.tensor([], dtype=torch.long, device=target_device)
    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=128).to(target_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if len(neg_pairs) == 0:
            pos_out, _, _ = model(adj_indices, adj_values, num_nodes, node_features,
                                 pos_u, pos_v, pos_u[:1], pos_v[:1])
            loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
        else:
            pos_out, neg_out, _ = model(adj_indices, adj_values, num_nodes, node_features,
                                         pos_u, pos_v, neg_u, neg_v)
            pos_loss = criterion(pos_out, torch.ones_like(pos_out))
            neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
            loss = 0.5 * (pos_loss + neg_loss)
        loss.backward()
        optimizer.step()
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(adj_indices, adj_values, num_nodes, node_features,
                                       pos_u[:1], pos_v[:1], neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1],
                                       neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1])
    return model, final_embeddings.cpu(), adj_indices.cpu(), adj_values.cpu()

# ==========================================
# RESEARCH DIRECTION SCORING
# ==========================================
def compute_research_direction_scores(model, node_features, final_emb, nx_graph,
                                       valid_concepts, concept_properties, ridge,
                                       embed_model, n_samples: int = 5000) -> pd.DataFrame:
    n_concepts = len(valid_concepts)
    if n_concepts < 3:
        return pd.DataFrame()
    u_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 5))
    v_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 5))
    candidate_pairs = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx:
            continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            continue
        candidate_pairs.append((u_idx, v_idx, u_c, v_c))
    if not candidate_pairs:
        return pd.DataFrame()
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pair_features = torch.cat([final_emb[u_tensor], final_emb[v_tensor]], dim=1)
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).numpy()
    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
    cos_sims = np.sum(emb_np[u_tensor.numpy()] * emb_np[v_tensor.numpy()], axis=1)
    results = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                expected_improvement = float(ridge.predict([[p_u, p_v, 1.0]])[0])
            except:
                expected_improvement = max(p_u, p_v) * 1.05
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = np.clip((expected_improvement - 50) / 200, 0, 1) if expected_improvement > 0 else 0
        D_uv = (alpha['gnn'] * gnn_scores[i] + alpha['novelty'] * semantic_novelty +
                alpha['gain'] * norm_gain + alpha['feas'] * (1.0 - feasibility))
        results.append({
            'concept_u': u_c, 'concept_v': v_c, 'gnn_affinity': float(gnn_scores[i]),
            'semantic_novelty': float(semantic_novelty), 'expected_property_gain': expected_improvement,
            'feasibility_score': float(feasibility), 'composite_score': float(D_uv)
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.head(min(100, len(df)))

# ==========================================
# MATHEMATICAL VALIDATION
# ==========================================
def validate_graph_metrics(nx_graph: nx.Graph, valid_concepts: List[str]) -> Dict[str, Any]:
    metrics = {}
    if nx_graph.number_of_nodes() < 3:
        return metrics
    try:
        from networkx.algorithms import community
        partition = list(community.greedy_modularity_communities(nx_graph))
        metrics["modularity"] = community.modularity(nx_graph, partition)
        metrics["n_communities"] = len(partition)
    except Exception:
        metrics["modularity"] = 0.0
        metrics["n_communities"] = 0
    try:
        embed_model = load_embedding_model()
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        if len(valid_concepts) >= 3:
            labels = np.zeros(len(valid_concepts))
            for i, c in enumerate(valid_concepts):
                for idx, comm in enumerate(partition if 'partition' in locals() else [[]]):
                    if c in comm:
                        labels[i] = idx
                        break
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
        else:
            metrics["silhouette_score"] = 0.0
    except Exception:
        metrics["silhouette_score"] = 0.0
    weights = [d.get('weight', 1) for _, _, d in nx_graph.edges(data=True)]
    if len(weights) > 10:
        p_values = []
        for w in weights[:50]:
            permuted = np.random.permutation(weights)
            p_values.append(np.sum(permuted >= w) / len(weights))
        metrics["edge_significance_p_mean"] = float(np.mean(p_values))
        metrics["edge_significant_count"] = int(sum(1 for p in p_values if p < 0.05))
    else:
        metrics["edge_significance_p_mean"] = 1.0
        metrics["edge_significant_count"] = 0
    try:
        metrics["avg_betweenness"] = np.mean(list(nx.betweenness_centrality(nx_graph).values()))
        metrics["avg_closeness"] = np.mean(list(nx.closeness_centrality(nx_graph).values()))
    except Exception:
        pass
    # v3.0: Edge type distribution
    edge_types = [d.get('edge_type', 'unknown') for _, _, d in nx_graph.edges(data=True)]
    metrics["edge_type_distribution"] = dict(Counter(edge_types))
    inferred_count = sum(1 for _, _, d in nx_graph.edges(data=True) if d.get('inferred', False))
    metrics["inferred_edges_count"] = inferred_count
    metrics["observed_edges_count"] = nx_graph.number_of_edges() - inferred_count
    return metrics

@st.cache_data(ttl=3600)
def compute_bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05):
    if len(scores) < 2:
        return float(np.mean(scores)), 0.0, 0.0
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(scores)), float(ci_low), float(ci_high)

# ==========================================
# v3.0: ENHANCED VISUALIZATION FUNCTIONS
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
# v3.0: PYVIS WITH N1/N2 ABBREVIATED LABELS & EDGE INSPECTION
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

    # v3.1: Floating edge info panel JavaScript
    edge_panel_js = """
    <script>
    (function() {
    var checkExist = setInterval(function() {
    if (typeof network !== 'undefined' && network !== null && network.body && network.body.data) {
    clearInterval(checkExist);
    var nodesDS = network.body.data.nodes;
    var edgesDS = network.body.data.edges;
    var savedNodeColors = {};
    var activeNodeId = null;

    function resetAll() {
        var nodeRestores = [];
        for (var nid in savedNodeColors) {
            nodeRestores.push({id: nid, color: savedNodeColors[nid]});
        }
        if (nodeRestores.length > 0) nodesDS.update(nodeRestores);
        savedNodeColors = {};
        activeNodeId = null;
        var panel = document.getElementById('edge-info-panel');
        if (panel) panel.style.display = 'none';
    }

    function showEdgeInfoPanel(nodeId, edgeList) {
        var panel = document.getElementById('edge-info-panel');
        if (!panel) {
            panel = document.createElement('div');
            panel.id = 'edge-info-panel';
            document.body.appendChild(panel);
        }
        panel.style.cssText = [
            'position:fixed', 'top:110px', 'right:24px', 'width:340px',
            'max-height:520px', 'overflow-y:auto', 'z-index:9999',
            'background:rgba(255,255,255,0.98)', 'border:2px solid #FFD700',
            'border-radius:12px', 'padding:14px 16px',
            'font-family:Inter,Segoe UI,Roboto,sans-serif',
            'box-shadow:0 10px 40px rgba(0,0,0,0.22)',
            'backdrop-filter:blur(6px)'
        ].join(';');

        var nodeData = nodesDS.get(nodeId);
        var nodeName = nodeData ? (nodeData.title ?
            (new DOMParser().parseFromString(nodeData.title,'text/html').body.textContent || nodeData.label || nodeId)
            : (nodeData.label || nodeId)) : nodeId;
        nodeName = nodeName.replace(/<[^>]*>/g,'').trim().split('\n')[0];

        edgeList.sort(function(a,b){ return parseFloat(b.weight)-parseFloat(a.weight); });

        var html = '<div style="font-size:14px;font-weight:700;color:#D32F2F;margin-bottom:8px;'
                 + 'border-bottom:2px solid #FFD700;padding-bottom:6px;">'
                 + '🔗 <span style="color:#1e293b;">' + nodeName + '</span> '
                 + '<span style="color:#64748b;font-weight:400;font-size:12px;">('
                 + edgeList.length + ' connections)</span></div>'
                 + '<div style="font-size:10.5px;color:#64748b;margin-bottom:8px;'
                 + 'font-style:italic;">Format: <b style="color:#D32F2F;">w</b> '
                 + 'N<sub>i</sub> ↔ N<sub>j</sub> = value (type)</div>';

        edgeList.forEach(function(e, idx){
            var typeColor = e.isInferred ? '#8b5cf6' : '#0ea5e9';
            html += '<div style="padding:6px 8px;margin-bottom:4px;background:#f8fafc;'
                  + 'border-left:3px solid ' + typeColor + ';border-radius:4px;'
                  + 'font-size:11.5px;line-height:1.5;">'
                  + '<span style="color:#94a3b8;font-size:10px;">' + (idx+1) + '.</span> '
                  + '<b style="color:#D32F2F;">w</b> '
                  + '<span style="color:#1e293b;">' + e.from + '</span>'
                  + ' <span style="color:#94a3b8;">↔</span> '
                  + '<span style="color:#1e293b;">' + e.to + '</span>'
                  + ' <span style="color:#64748b;">=</span> '
                  + '<b style="color:#0ea5e9;">' + e.weight + '</b> '
                  + '<span style="color:' + typeColor + ';font-size:10px;font-weight:600;">'
                  + '(' + e.type + ')</span></div>';
        });

        panel.innerHTML = html;
        panel.style.display = 'block';
    }

    network.on("selectNode", function(params) {
        var nodeId = params.nodes[0];
        if (activeNodeId !== null && activeNodeId !== nodeId) resetAll();
        activeNodeId = nodeId;

        var connectedEdges = network.getConnectedEdges(nodeId);
        var connectedNodes = network.getConnectedNodes(nodeId);

        var nodeUpdates = [];
        connectedNodes.forEach(function(nId){
            var n = nodesDS.get(nId);
            if (n && !savedNodeColors[nId]) {
                savedNodeColors[nId] = JSON.parse(JSON.stringify(n.color));
                var newColor = JSON.parse(JSON.stringify(n.color));
                if (typeof newColor === 'string') {
                    newColor = {background:newColor, border:'#FFD700'};
                } else {
                    newColor.border = '#FFD700';
                    if (newColor.highlight) newColor.highlight.border = '#FFD700';
                    if (newColor.hover) newColor.hover.border = '#FFD700';
                }
                nodeUpdates.push({id:nId, color:newColor});
            }
        });
        if (nodeUpdates.length > 0) nodesDS.update(nodeUpdates);

        var edgeInfoList = [];
        connectedEdges.forEach(function(eId){
            var e = edgesDS.get(eId);
            if (!e) return;
            var fromNode = nodesDS.get(e.from);
            var toNode = nodesDS.get(e.to);
            var fromLabel = fromNode ? (fromNode.label || e.from) : e.from;
            var toLabel = toNode ? (toNode.label || e.to) : e.to;
            fromLabel = String(fromLabel).replace(/<[^>]*>/g,'').trim();
            toLabel = String(toLabel).replace(/<[^>]*>/g,'').trim();

            var w = (typeof e.value === 'number') ? e.value : (e.width || 1);
            var edgeType = 'unknown', isInferred = false;
            if (e.title) {
                var tmp = document.createElement('div');
                tmp.innerHTML = e.title;
                var desc = (tmp.textContent || tmp.innerText || '').replace(/\s+/g,' ').trim();
                var m = desc.match(/Type:\s*(\w+)/i);
                if (m) edgeType = m[1];
                if (desc.indexOf('Inferred: true') !== -1) isInferred = true;
            }
            edgeInfoList.push({
                from: fromLabel, to: toLabel,
                weight: (typeof w === 'number') ? w.toFixed(2) : w,
                type: edgeType, isInferred: isInferred
            });
        });

        showEdgeInfoPanel(nodeId, edgeInfoList);
    });

    network.on("deselectNode", function(){ resetAll(); });
    network.on("click", function(params){
        if (params.nodes.length === 0 && activeNodeId !== null) resetAll();
    });
    }
    }, 250);
    })();
    </script>
    """
    html_content = html_content.replace("</body>", edge_panel_js + "</body>")
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

def render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name="viridis",
                            custom_labels=None, top_n_nodes=0, node_label_size=10,
                            theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos = nx.spring_layout(nx_graph, k=1.5, iterations=50, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x, edge_y, edge_hover = [], [], []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)
        confidence = nx_graph[u][v].get('confidence', 1.0)
        edge_hover.extend([f"<b>{u} ↔ {v}</b><br>Weight: {w:.2f}<br>Type: {edge_type}<br>Inferred: {is_inferred}<br>Confidence: {confidence:.2f}"] * 2 + [None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=1, color=theme['edge_unknown']),
                            hoverinfo='text', hovertext=edge_hover, name='Connections')
    node_x, node_y, node_text, node_size, node_color, node_labels = [], [], [], [], [], []
    for i, node in enumerate(nx_graph.nodes()):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}<br>Type: {concept_type}")
        node_size.append(max(8, min(35, deg * 2.5 + 10)))
        node_color.append(cmap_colors[i])
        node_labels.append(custom_labels.get(node, node) if custom_labels else node)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            marker=dict(size=node_size, color=node_color,
                                       line=dict(width=2, color=theme['node_border'])),
                            text=node_labels, textposition="bottom center",
                            textfont=dict(size=node_label_size, color=theme['font']),
                            hovertext=node_text, hoverinfo='text', name='Concepts')
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=0),
                                     plot_bgcolor=theme['plotly_bg'], paper_bgcolor=theme['plotly_paper'],
                                     font=dict(color=theme['font']),
                                     xaxis=dict(showgrid=True, gridcolor=theme['grid_color'],
                                                zeroline=False, showticklabels=False, linecolor=theme['axis_color']),
                                     yaxis=dict(showgrid=True, gridcolor=theme['grid_color'],
                                                zeroline=False, showticklabels=False, linecolor=theme['axis_color'])))
    st.plotly_chart(fig, use_container_width=True)


def render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name="viridis", top_n_nodes=0,
                            theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(nx_graph.nodes()) < 3:
        st.info("3D view requires ≥3 nodes.")
        return
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos_3d = nx.spring_layout(nx_graph, dim=3, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x, edge_y, edge_z = [], [], []
    for u, v in nx_graph.edges():
        x0, y0, z0 = pos_3d[u]; x1, y1, z1 = pos_3d[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None]); edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
                              line=dict(width=2, color=theme['edge_unknown']), hoverinfo='skip')
    node_x, node_y, node_z, node_text, node_size, node_color, node_labels = [], [], [], [], [], [], []
    for i, node in enumerate(nx_graph.nodes()):
        x, y, z = pos_3d[node]
        node_x.append(x); node_y.append(y); node_z.append(z)
        deg = nx_graph.degree(node); freq = len(concept_abstract_map.get(node, []))
        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}<br>Type: {concept_type}")
        node_size.append(max(6, min(25, deg * 2 + 8)))
        node_color.append(cmap_colors[i])
        node_labels.append(node)
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text',
                                marker=dict(size=node_size, color=node_color, opacity=0.9),
                                text=node_labels, textposition="top center",
                                textfont=dict(size=8, color=theme['font']),
                                hovertext=node_text, hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(scene=dict(xaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color']),
                                                 yaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color']),
                                                 zaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color'])),
                                     margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
                                     paper_bgcolor=theme['plotly_paper']))
    st.plotly_chart(fig, use_container_width=True)


def render_graph_fallback(nx_graph, concept_abstract_map, theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    st.markdown(f"### 📊 Graph Summary (Text View)")
    st.markdown(f"- **Nodes**: {len(nx_graph.nodes())}")
    st.markdown(f"- **Edges**: {len(nx_graph.edges())}")

    # v3.0: Edge type breakdown
    edge_types = Counter(nx_graph[u][v].get('edge_type', 'unknown') for u, v in nx_graph.edges())
    st.markdown("**Edge Type Distribution:**")
    for etype, count in edge_types.most_common():
        st.markdown(f"  - {etype}: {count}")

    inferred_count = sum(1 for u, v in nx_graph.edges() if nx_graph[u][v].get('inferred', False))
    st.markdown(f"- **Inferred Edges**: {inferred_count}")
    st.markdown(f"- **Observed Edges**: {len(nx_graph.edges()) - inferred_count}")

    if len(nx_graph.edges()) > 0:
        edge_list = [(u, v, nx_graph[u][v].get('weight', 1), nx_graph[u][v].get('edge_type', 'unknown'),
                     nx_graph[u][v].get('inferred', False)) for u, v in nx_graph.edges()]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        st.markdown("**🔗 Top 20 Strongest Connections:**")
        for i, (u, v, w, etype, inferred) in enumerate(edge_list[:20], 1):
            marker = "🤖" if inferred else "📄"
            st.markdown(f"{marker} {i}. `{u}` ↔ `{v}` (weight: {w:.2f}, type: {etype})")
    if len(concept_abstract_map) > 0:
        freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in nx_graph.nodes()]
        freq_data.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**📈 Top Concepts by Frequency:**")
        st.dataframe(pd.DataFrame(freq_data[:15], columns=["Concept", "Abstract Count"]), use_container_width=True)

# ==========================================
# v3.0: SYMBOL-BASED SUNBURST CHART
# ==========================================
def build_category_hierarchy(valid_concepts: List[str], concept_abstract_map: Dict, top_n_per_category: int = 40):
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    category_map = abstract_concepts_to_categories(valid_concepts)
    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
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
# v3.0: EXTRA VISUALIZATIONS
# ==========================================
def render_radar_chart(concept_scores_df: pd.DataFrame, top_k: int = 15, cmap_name: str = "viridis", theme=None):
    if concept_scores_df.empty or len(concept_scores_df) < 2:
        st.info("Not enough concepts for radar chart.")
        return
    metrics = ['frequency', 'semantic_density', 'coherence_score', 'distillation_efficiency']
    available_metrics = [m for m in metrics if m in concept_scores_df.columns]
    if not available_metrics:
        st.warning("No metrics available for radar chart.")
        return
    top_concepts = concept_scores_df.nlargest(top_k, 'distillation_efficiency')
    normalized = top_concepts.copy()
    for m in available_metrics:
        col = normalized[m]
        if col.max() > col.min():
            normalized[m] = (col - col.min()) / (col.max() - col.min())
        else:
            normalized[m] = 0.5
    categories = available_metrics
    fig = go.Figure()
    colors = get_colormap_colors(cmap_name, len(normalized))
    for idx, (_, row) in enumerate(normalized.iterrows()):
        concept = row['concept']
        values = [row[m] for m in categories]
        values += values[:1]
        angles = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill='toself', name=concept[:20],
            line=dict(width=2, color=colors[idx]), fillcolor=colors[idx], opacity=0.6
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Top Concepts: Multi-Dimensional Comparison",
        showlegend=True, width=750, height=600,
        paper_bgcolor=theme["plotly_paper"] if theme else "#ffffff",
        font=dict(color=theme["font"] if theme else "#000000"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)


# v3.0: t-SNE Projection
def render_tsne_projection(valid_concepts, concept_abstract_map, embed_model, cmap_name="viridis"):
    """Render 2D t-SNE projection of concept embeddings, colored by category."""
    if len(valid_concepts) < 3:
        st.info("Need at least 3 concepts for t-SNE projection.")
        return

    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
    perplexity = min(30, len(valid_concepts) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    freqs = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
    categories = [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]

    df_tsne = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'concept': valid_concepts,
        'frequency': freqs,
        'category': categories
    })

    fig = px.scatter(
        df_tsne, x='x', y='y', 
        size='frequency', 
        color='category',
        hover_name='concept',
        title="t-SNE Semantic Clustering of Concepts",
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        width=800, height=600,
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("💡 **Interpretation:** Concepts closer together are semantically similar. Size = frequency in corpus.")


# v3.0: Keyword Burst Detection
def detect_keyword_bursts(df, valid_concepts, concept_abstract_map):
    """Detect temporal spikes in concept frequency (bursts)."""
    if "Year" not in df.columns:
        st.info("No 'Year' column found for temporal analysis.")
        return pd.DataFrame()

    burst_data = []
    for concept in valid_concepts[:100]:  # Limit for speed
        doc_indices = concept_abstract_map.get(concept, [])
        years = []
        for idx in doc_indices:
            if idx < len(df) and pd.notna(df.iloc[idx].get("Year")):
                years.append(int(df.iloc[idx]["Year"]))

        if len(years) < 3:
            continue

        year_counts = Counter(years)
        if not year_counts:
            continue

        avg_freq = np.mean(list(year_counts.values()))
        max_freq = max(year_counts.values())
        max_burst = max_freq / max(1, avg_freq)

        if max_burst > 1.5:  # Threshold for burst
            burst_data.append({
                "concept": concept, 
                "burst_score": round(max_burst, 2), 
                "peak_year": max(year_counts, key=year_counts.get),
                "peak_count": max_freq,
                "total_papers": len(years)
            })

    # Prevent KeyError when no keyword bursts are detected
    if not burst_data:
        return pd.DataFrame()
    return pd.DataFrame(burst_data).sort_values("burst_score", ascending=False)


def render_burst_chart(burst_df, top_k=20):
    """Render keyword burst visualization."""
    if burst_df.empty:
        st.info("No significant keyword bursts detected.")
        return

    display_df = burst_df.head(top_k)
    fig = px.bar(
        display_df, 
        x='burst_score', 
        y='concept', 
        color='peak_year',
        orientation='h',
        title=f"🔥 Top {len(display_df)} Keyword Bursts (Temporal Spikes)",
        labels={'burst_score': 'Burst Score', 'concept': 'Concept'},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=max(400, len(display_df) * 25))
    st.plotly_chart(fig, use_container_width=True)


# v3.0: Co-occurrence Heatmap
def render_cooccurrence_heatmap(nx_graph, valid_concepts, top_n=30):
    """Render dense co-occurrence matrix for top concepts."""
    if len(valid_concepts) < 3:
        st.info("Need more concepts for heatmap.")
        return

    # Select top concepts by degree
    degrees = dict(nx_graph.degree(weight='weight'))
    top_concepts = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n]

    # Build co-occurrence matrix
    n = len(top_concepts)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(top_concepts):
        for j, c2 in enumerate(top_concepts):
            if i == j:
                matrix[i][j] = nx_graph.nodes[c1].get('frequency', 0)
            elif nx_graph.has_edge(c1, c2):
                matrix[i][j] = nx_graph[c1][c2].get('cooccurrence', 0)

    fig = px.imshow(
        matrix,
        x=top_concepts,
        y=top_concepts,
        color_continuous_scale='YlOrRd',
        title=f"Co-occurrence Matrix (Top {n} Concepts)",
        labels=dict(color="Co-occurrence Count")
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=700, width=700
    )
    st.plotly_chart(fig, use_container_width=True)


# v3.0: Network Motif Analysis
def render_motif_analysis(nx_graph):
    """Analyze and visualize network motifs (triangles, cliques, stars)."""
    if nx_graph.number_of_nodes() < 3:
        st.info("Need at least 3 nodes for motif analysis.")
        return

    motifs = {}

    # Triangles
    try:
        triangles = nx.triangles(nx_graph)
        motifs['triangles'] = sum(triangles.values()) // 3
    except:
        motifs['triangles'] = 0

    # Cliques
    try:
        cliques = list(nx.find_cliques(nx_graph))
        motifs['cliques_3'] = sum(1 for c in cliques if len(c) == 3)
        motifs['cliques_4'] = sum(1 for c in cliques if len(c) == 4)
        motifs['max_clique_size'] = max(len(c) for c in cliques) if cliques else 0
    except:
        motifs['cliques_3'] = motifs['cliques_4'] = motifs['max_clique_size'] = 0

    # Star motifs (hubs with high degree)
    degrees = dict(nx_graph.degree())
    star_threshold = np.percentile(list(degrees.values()), 90)
    star_nodes = [n for n, d in degrees.items() if d >= star_threshold]
    motifs['star_nodes'] = len(star_nodes)

    # Clustering
    try:
        motifs['avg_clustering'] = nx.average_clustering(nx_graph)
    except:
        motifs['avg_clustering'] = 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Triangles", motifs['triangles'])
    col2.metric("3-Cliques", motifs['cliques_3'])
    col3.metric("4-Cliques", motifs['cliques_4'])
    col4.metric("Star Hubs", motifs['star_nodes'])

    st.metric("Average Clustering", f"{motifs['avg_clustering']:.3f}")

    # Star hub visualization
    if star_nodes:
        st.markdown("**⭐ Star Hub Concepts (Top 10% by Degree):**")
        star_data = [(n, degrees[n], nx_graph.nodes[n].get('concept_type', 'general')) 
                     for n in star_nodes[:10]]
        star_df = pd.DataFrame(star_data, columns=['Concept', 'Degree', 'Type'])
        st.dataframe(star_df, use_container_width=True)


# v3.0: Temporal Trend Analysis
def render_temporal_trends(df, valid_concepts, concept_abstract_map, top_k=10):
    """Render temporal trend lines for top concepts."""
    if "Year" not in df.columns:
        st.info("No 'Year' column available for temporal analysis.")
        return

    # Select top concepts
    degrees = {c: len(concept_abstract_map.get(c, [])) for c in valid_concepts}
    top_concepts = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_k]

    trend_data = []
    for concept in top_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        years = []
        for idx in doc_indices:
            if idx < len(df) and pd.notna(df.iloc[idx].get("Year")):
                years.append(int(df.iloc[idx]["Year"]))

        if years:
            year_counts = Counter(years)
            for year, count in sorted(year_counts.items()):
                trend_data.append({
                    'concept': concept,
                    'year': year,
                    'count': count
                })

    if not trend_data:
        st.info("No temporal data available.")
        return

    trend_df = pd.DataFrame(trend_data)
    fig = px.line(
        trend_df, 
        x='year', 
        y='count', 
        color='concept',
        title=f"📈 Temporal Trends of Top {len(top_concepts)} Concepts",
        labels={'year': 'Year', 'count': 'Paper Count', 'concept': 'Concept'}
    )
    fig.update_layout(
        xaxis_type='linear',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


# v3.0: Semantic Drift Detection
def detect_semantic_drift(df, valid_concepts, concept_abstract_map, embed_model):
    """Detect if concept meanings have shifted over time."""
    if "Year" not in df.columns or len(df) < 10:
        return pd.DataFrame()

    # Split into early and late papers
    median_year = df["Year"].median()

    drift_data = []
    for concept in valid_concepts[:50]:  # Limit for speed
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 4:
            continue

        early_texts = []
        late_texts = []
        for idx in doc_indices:
            if idx < len(df):
                year = df.iloc[idx].get("Year")
                text = str(df.iloc[idx].get("abstract", df.iloc[idx].get("title", "")))
                if pd.notna(year) and text:
                    if year <= median_year:
                        early_texts.append(text)
                    else:
                        late_texts.append(text)

        if len(early_texts) >= 2 and len(late_texts) >= 2:
            early_emb = embed_model.encode(early_texts, show_progress_bar=False)
            late_emb = embed_model.encode(late_texts, show_progress_bar=False)

            early_centroid = np.mean(early_emb, axis=0)
            late_centroid = np.mean(late_emb, axis=0)

            drift = 1 - cosine_similarity([early_centroid], [late_centroid])[0][0]

            if drift > 0.1:  # Significant drift threshold
                drift_data.append({
                    'concept': concept,
                    'drift_score': round(drift, 3),
                    'median_year': int(median_year),
                    'early_papers': len(early_texts),
                    'late_papers': len(late_texts)
                })

    # Prevent KeyError when no concepts meet the drift threshold
    if not drift_data:
        return pd.DataFrame()
    return pd.DataFrame(drift_data).sort_values("drift_score", ascending=False)


# ==========================================
# v3.0: REASONING DASHBOARD
# ==========================================
def render_reasoning_dashboard(nx_graph, valid_concepts, ontology, graph_builder):
    """v3.0: Comprehensive reasoning and inference insights dashboard."""
    st.subheader("🔍 Reasoning & Inference Insights")

    # 1. Concept Type Distribution
    st.markdown("**📊 Concept Type Distribution:**")
    types = [nx_graph.nodes[c].get('concept_type', 'general') for c in valid_concepts]
    type_dist = Counter(types)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=list(type_dist.values()), 
            names=list(type_dist.keys()), 
            title="Concept Types",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=list(type_dist.keys()), 
            y=list(type_dist.values()),
            title="Concept Type Counts",
            labels={'x': 'Type', 'y': 'Count'},
            color=list(type_dist.keys()),
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2. Edge Type Distribution
    st.markdown("**🔗 Edge Type Distribution (Observed vs. Inferred):**")
    edge_types = [nx_graph[u][v].get('edge_type', 'unknown') for u, v in nx_graph.edges()]
    type_counts = Counter(edge_types)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=list(type_counts.values()), 
            names=list(type_counts.keys()), 
            title="Edge Types",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Inferred vs Observed
        inferred_count = sum(1 for u, v in nx_graph.edges() if nx_graph[u][v].get('inferred', False))
        observed_count = nx_graph.number_of_edges() - inferred_count
        fig = px.pie(
            values=[observed_count, inferred_count],
            names=['Observed', 'Inferred'],
            title="Observed vs. Inferred Edges",
            color_discrete_map={'Observed': '#4CAF50', 'Inferred': '#FF9800'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3. Inferred Causal Chains
    st.markdown("**🧠 Inferred Cross-Domain Bridges (Process → Microstructure → Property):**")
    if graph_builder and hasattr(graph_builder, 'reasoning_paths') and graph_builder.reasoning_paths:
        paths_df = pd.DataFrame([
            {"Chain": " → ".join(p), "Length": len(p)} 
            for p in graph_builder.reasoning_paths[:20]
        ])
        st.dataframe(paths_df, use_container_width=True)

        # Visualize as flow diagram
        fig = px.funnel(
            paths_df.head(10), 
            x='Length', 
            y='Chain',
            title="Top Inferred Reasoning Chains"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cross-domain bridges inferred. Enable 'USE_INFERENCE' in config.")

    # 4. Synonym Resolution Log
    st.markdown("**📝 Synonym Resolution Examples:**")
    if ontology and ontology.synonym_to_canonical:
        sample_resolutions = list(ontology.synonym_to_canonical.items())[:20]
        res_df = pd.DataFrame(sample_resolutions, columns=["Raw Term", "Canonical Form"])
        st.dataframe(res_df, use_container_width=True)

    # 5. Hierarchical Relationships
    st.markdown("**🌳 Hierarchical Relationships (is-a chains):**")
    hierarchy_examples = []
    for concept in valid_concepts[:20]:
        hypernyms = ontology.get_hypernyms(concept) if ontology else set()
        if hypernyms:
            hierarchy_examples.append({
                'concept': concept,
                'hypernyms': ' → '.join(list(hypernyms)[:3])
            })
    if hierarchy_examples:
        hier_df = pd.DataFrame(hierarchy_examples)
        st.dataframe(hier_df, use_container_width=True)
    else:
        st.info("No hierarchical relationships found for displayed concepts.")


# ==========================================
# EXPORT FUNCTIONS
# ==========================================
def export_graph(nx_graph, concept_abstract_map, format_type: str):
    if format_type == "GraphML":
        try:
            nx.write_graphml_lxml(nx_graph, "nano_graph_v3.graphml")
        except:
            nx.write_graphml(nx_graph, "nano_graph_v3.graphml")
        with open("nano_graph_v3.graphml", "rb") as f:
            return f.read(), "application/graphml+xml", "nano_graph_v3.graphml"
    elif format_type == "JSON":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "nano_graph_v3.json"
    elif format_type == "CSV (Edges)":
        edge_data = []
        for u, v, data in nx_graph.edges(data=True):
            row = {"source": u, "target": v}
            row.update({k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))})
            edge_data.append(row)
        csv_df = pd.DataFrame(edge_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "nano_edges_v3.csv"
    elif format_type == "CSV (Nodes)":
        node_data = []
        for node in nx_graph.nodes():
            row = {"concept": node, "frequency": len(concept_abstract_map.get(node, [])),
                   "degree": nx_graph.degree(node),
                   "concept_type": nx_graph.nodes[node].get('concept_type', 'general')}
            row.update({k: v for k, v in nx_graph.nodes[node].items()})
            node_data.append(row)
        csv_df = pd.DataFrame(node_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "nano_nodes_v3.csv"
    elif format_type == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [get_nanomaterials_category_color(n) for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                   node_size=400, font_size=7, font_weight='bold', edgecolors='white', linewidths=1)
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buf.seek(0); plt.close()
            return buf.read(), "image/png", "nano_graph_v3.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    return None, None, None

# ==========================================
# GRAPH METRICS DASHBOARD
# ==========================================
def compute_graph_metrics(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "clustering": nx.average_clustering(G) if G.number_of_nodes() > 2 else 0,
        "connected_components": nx.number_connected_components(G),
        "avg_clustering": nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
    }
    try:
        bc = nx.betweenness_centrality(G, normalized=True, k=min(100, G.number_of_nodes()))
        top_bridges = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics["top_bridges"] = top_bridges
        metrics["avg_betweenness"] = np.mean(list(bc.values()))
    except Exception:
        metrics["top_bridges"] = []
    return metrics

def display_metric_dashboard(metrics: dict, theme=None):
    if not metrics:
        st.warning("No graph metrics available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", metrics["nodes"]); col2.metric("Edges", metrics["edges"])
    col3.metric("Density", f"{metrics['density']:.3f}"); col4.metric("Avg Degree", f"{metrics['avg_degree']:.2f}")
    col5, col6, col7 = st.columns(3)
    col5.metric("Clustering", f"{metrics['clustering']:.3f}")
    col6.metric("Components", metrics["connected_components"])
    col7.metric("Avg Betweenness", f"{metrics.get('avg_betweenness', 0):.3f}")
    if metrics.get("top_bridges"):
        st.markdown("**🌉 Top Bridge Concepts (High Betweenness)**")
        bridge_df = pd.DataFrame(metrics["top_bridges"], columns=["Concept", "Bridge Score"])
        st.dataframe(bridge_df, use_container_width=True)

# ==========================================
# THEME CONFIGURATION
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
# SIDEBAR CONFIGURATION
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
# MAIN APPLICATION
# ==========================================
def main():
    st.title("🔬 Nanomaterials-ConceptGraph v3.0: Deep Semantic Reasoning Engine")
    st.caption("Ontology-aware reasoning • Embedding-based equivalence • Hierarchical taxonomy • Cross-domain inference • Cause-effect extraction • N1/N2 annotations • Symbol Sunburst")
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
                config["USE_INFERENCE"] = st.session_state.get('use_inference', True)
                config["USE_CAUSAL_EXTRACTION"] = st.session_state.get('use_causal_extraction', True)
                st.write(f"📊 Adaptive config: {config}")
                progress_bar.progress(0.15)

                # v3.0: Initialize rich ontology
                st.write("📚 Building nanomaterials ontology...")
                ontology = NanomaterialsOntology()
                ontology.build_embeddings(embed_model)
                st.success(f"✅ Ontology loaded: {len(ontology.concepts)} concepts, {len(ontology.synonym_to_canonical)} synonyms")
                progress_bar.progress(0.20)

                # v3.0: Initialize resolver with disambiguation
                resolver = AdvancedConceptResolver(
                    ontology, embed_model, 
                    similarity_threshold=st.session_state.get('sim_threshold', 0.85)
                )
                extractor = EnhancedConceptExtractor(ontology, resolver)

                # v3.0: Parallel extraction with relationships
                st.write("🔍 Parallel extraction with cause-effect detection...")
                n_workers = st.session_state.get('parallel_workers', 8)
                all_concepts, all_metrics, all_relationships = extract_concepts_parallel(
                    df_filtered, selected_text_cols, extractor, max_workers=n_workers
                )
                st.write(f"✅ Extracted raw concepts from {len(all_concepts)} documents")
                st.write(f"✅ Found {len(all_relationships)} cause-effect relationships")
                progress_bar.progress(0.35)

                # v3.0: Global resolution
                st.write("🚀 Performing global semantic resolution via BLAS...")
                resolved_map = extractor.finalize_global_resolution()
                for doc_idx in range(len(all_concepts)):
                    all_concepts[doc_idx] = [resolved_map.get(c, c) for c in all_concepts[doc_idx]]
                st.success(f"✅ Resolved {len(resolved_map)} unique phrases globally")
                progress_bar.progress(0.45)

                st.write("🧹 Filtering and normalizing concepts...")
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(
                    all_concepts, config, ontology
                )
                st.write(f"✅ **{len(valid_concepts)}** valid concepts retained")
                progress_bar.progress(0.55)

                if len(valid_concepts) < 5:
                    st.error("Too few concepts extracted. Try lowering frequency thresholds.")
                    return

                # v3.0: Build reasoning-enhanced graph
                st.write("🕸️ Building reasoning-enhanced concept graph...")
                builder = ReasoningEnhancedGraphBuilder(ontology, extractor)
                nx_graph = builder.build_graph(
                    all_concepts, valid_concepts, concept_to_id,
                    relationships=all_relationships,
                    embed_model=embed_model, config=config
                )

                # Report edge statistics
                edge_types = Counter(nx_graph[u][v].get('edge_type', 'unknown') for u, v in nx_graph.edges())
                inferred_count = sum(1 for u, v in nx_graph.edges() if nx_graph[u][v].get('inferred', False))
                st.write(f"✅ Graph: {len(valid_concepts)} nodes, {nx_graph.number_of_edges()} edges")
                st.write(f"   - Co-occurrence: {edge_types.get('cooccurrence', 0)}")
                st.write(f"   - Semantic: {edge_types.get('semantic', 0)}")
                st.write(f"   - Hierarchical: {edge_types.get('hypernym', 0) + edge_types.get('hyponym', 0)}")
                st.write(f"   - Causal: {edge_types.get('causes', 0) + edge_types.get('influences', 0)}")
                st.write(f"   - Bridge (inferred): {edge_types.get('bridge', 0)}")
                st.write(f"   - Total inferred: {inferred_count}")
                progress_bar.progress(0.65)

                try:
                    d_prev_dict = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=4))
                except Exception:
                    d_prev_dict = {}
                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, valid_concepts, concept_to_id, config)

                st.write("🧬 Generating node embeddings...")
                try:
                    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
                    node_features = torch.tensor(embeddings, dtype=torch.float32)
                except Exception:
                    node_features = torch.randn(len(valid_concepts), 384)
                st.write(f"✅ Node features: {node_features.shape}")
                progress_bar.progress(0.75)

                st.write("🤖 Training GraphSAGE...")
                def training_progress(epoch, loss):
                    progress = 0.75 + (epoch / 50) * 0.10
                    progress_bar.progress(min(1.0, progress))
                    if epoch % 10 == 0:
                        status.write(f"📊 Epoch {epoch}/50 | Loss: {loss:.4f}")
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, training_progress
                )
                st.success("✅ GNN training complete")
                progress_bar.progress(0.90)

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
                progress_bar.progress(0.95)

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
                    # v3.0: Store reasoning objects
                    "ontology": ontology,
                    "builder": builder,
                    "relationships": all_relationships,
                    "df": df_filtered
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

        # v3.0: Get reasoning objects
        ontology = data.get("ontology")
        builder = data.get("builder")
        df = data.get("df")

        # v3.0: Extended tabs with Reasoning Dashboard and Extra Analytics
        viz_tab, distill_tab, scores_tab, valid_tab, reasoning_tab, extra_tab, export_tab = st.tabs([
            "🎨 Visualization", "📊 Distillation", "🎯 Research Directions", 
            "📐 Validation", "🧠 Reasoning Dashboard", "📈 Extra Analytics", "📥 Export"
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
            layout_mode = st.session_state.get("layout_mode", "force-directed")
            physics = st.session_state.get('physics_enabled', True)
            physics_preset = st.session_state.get('effective_physics', PHYSICS_PRESETS["Stable (Default)"])
            theme = THEME_PRESETS.get(st.session_state.get('theme', 'Bright (Default)'), THEME_PRESETS["Bright (Default)"])

            top_n = st.session_state.get('top_n_graph', 0)

            # v3.0: Get visualization options
            use_abbreviated = st.session_state.get('use_abbreviated_labels', False)
            max_label_len = st.session_state.get('max_label_length', 15)
            edge_label_mode = st.session_state.get('edge_label_mode', 'hover')

            if viz_choice == "PyVis (Interactive)":
                render_graph_pyvis(
                    nx_graph, concept_abstract_map, physics_enabled=physics,
                    cmap_name=cmap, top_n_nodes=top_n,
                    theme=theme, physics_preset=physics_preset,
                    use_abbreviated_labels=use_abbreviated,
                    max_label_length=max_label_len,
                    edge_label_mode=edge_label_mode
                , layout_mode=layout_mode)
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

            with st.expander("📈 Domain Hierarchy (Symbol Sunburst)"):
                labels, parents, values = build_category_hierarchy(valid_concepts, concept_abstract_map,
                                                                    top_n_per_category=st.session_state.get('top_n_sunburst', 0))
                use_symbols = st.session_state.get('use_symbol_sunburst', True)
                render_sunburst_chart(labels, parents, values, cmap_name=cmap, theme=theme,
                                      use_symbols=use_symbols)

            with st.expander("📡 Concept Radar"):
                radar_k = st.session_state.get('top_n_radar', 15)
                if radar_k == 0:
                    radar_k = min(15, len(distill_df))
                render_radar_chart(distill_df, top_k=radar_k, cmap_name=cmap, theme=theme)

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
                                  file_name="research_directions_v3.csv", mime="text/csv")

        with valid_tab:
            st.subheader("📐 Mathematical Validation")
            val_metrics = validate_graph_metrics(nx_graph, valid_concepts)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Modularity", f"{val_metrics.get('modularity', 0):.3f}")
            col2.metric("Silhouette", f"{val_metrics.get('silhouette_score', 0):.3f}")
            col3.metric("Communities", val_metrics.get('n_communities', 0))
            col4.metric("Significant Edges", val_metrics.get('edge_significant_count', 0))

            # v3.0: Edge type distribution
            if "edge_type_distribution" in val_metrics:
                st.markdown("**🔗 Edge Type Distribution:**")
                edge_type_df = pd.DataFrame([
                    {"Type": k, "Count": v} 
                    for k, v in val_metrics["edge_type_distribution"].items()
                ])
                st.dataframe(edge_type_df, use_container_width=True)

            if "inferred_edges_count" in val_metrics:
                col1, col2 = st.columns(2)
                col1.metric("Inferred Edges", val_metrics["inferred_edges_count"])
                col2.metric("Observed Edges", val_metrics["observed_edges_count"])

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

        # v3.0: REASONING DASHBOARD TAB
        with reasoning_tab:
            if ontology and builder:
                render_reasoning_dashboard(nx_graph, valid_concepts, ontology, builder)
            else:
                st.info("Reasoning dashboard requires ontology data. Re-run analysis with reasoning enabled.")

        # v3.0: EXTRA ANALYTICS TAB
        with extra_tab:
            st.subheader("📈 Advanced Analytics Suite")

            # t-SNE Projection
            with st.expander("🎯 t-SNE Semantic Projection", expanded=True):
                st.markdown("**2D t-SNE projection of concept embeddings**")
                embed_model = data.get("embed_model")
                if embed_model:
                    render_tsne_projection(valid_concepts, concept_abstract_map, embed_model, cmap)

            # Keyword Burst Detection
            with st.expander("🔥 Keyword Burst Detection"):
                if df is not None and "Year" in df.columns:
                    burst_df = detect_keyword_bursts(df, valid_concepts, concept_abstract_map)
                    render_burst_chart(burst_df, top_k=20)
                else:
                    st.info("Temporal burst detection requires 'Year' column in data.")

            # Co-occurrence Heatmap
            with st.expander("🔥 Co-occurrence Heatmap"):
                render_cooccurrence_heatmap(nx_graph, valid_concepts, top_n=30)

            # Network Motif Analysis
            with st.expander("🕸️ Network Motif Analysis"):
                render_motif_analysis(nx_graph)

            # Temporal Trends
            with st.expander("📈 Temporal Trends"):
                if df is not None and "Year" in df.columns:
                    render_temporal_trends(df, valid_concepts, concept_abstract_map, top_k=10)
                else:
                    st.info("Temporal trends require 'Year' column in data.")

            # Semantic Drift
            with st.expander("🌊 Semantic Drift Detection"):
                if df is not None and "Year" in df.columns and embed_model:
                    with st.spinner("Analyzing semantic drift..."):
                        drift_df = detect_semantic_drift(df, valid_concepts, concept_abstract_map, embed_model)
                        if not drift_df.empty:
                            st.markdown("**Concepts with significant semantic drift over time:**")
                            st.dataframe(drift_df, use_container_width=True)
                            fig = px.bar(
                                drift_df.head(15), 
                                x='drift_score', 
                                y='concept',
                                orientation='h',
                                title="Semantic Drift Scores (Higher = More Drift)",
                                labels={'drift_score': 'Drift Score', 'concept': 'Concept'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No significant semantic drift detected.")
                else:
                    st.info("Semantic drift detection requires 'Year' column and embedding model.")

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
                'category': [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts],
                'concept_type': [nx_graph.nodes[c].get('concept_type', 'general') for c in valid_concepts]
            })
            csv_concepts = concept_list_df.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download Concept List (CSV)", data=csv_concepts,
                              file_name="concepts_v3.csv", mime="text/csv")

            # v3.0: Export relationships
            if "relationships" in data and data["relationships"]:
                rel_df = pd.DataFrame([r.to_dict() for r in data["relationships"]])
                csv_rels = rel_df.to_csv(index=False).encode('utf-8')
                st.download_button("🔗 Download Relationships (CSV)", data=csv_rels,
                                  file_name="relationships_v3.csv", mime="text/csv")

if __name__ == "__main__":
    main()
