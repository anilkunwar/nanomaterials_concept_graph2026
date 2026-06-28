#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
NanoGraph-Explorer: Advanced Concept Graph Analytics for Core-Shell Ag-Cu Nanostructures
================================================================================
A publication-ready, large-corpus concept graph extraction and visualization platform
for nanomaterials literature mining. Optimized for core-shell Ag-Cu nanostructures
with plasmonic, catalytic, and interfacial engineering focus.

Version: 2.0 (Enhanced Publication Edition)
Features:
- Robust multi-format data ingestion (JSON/JSONL/CSV/XML/BIB) with BOM handling
- Domain-specific concept extraction with semantic clustering and TF-IDF weighting
- Hybrid graph construction (co-occurrence + semantic similarity + temporal edges)
- GraphSAGE-powered GNN for research direction scoring
- Interactive graph editing with undo/redo, merge/split, filter operations
- Multi-scale analysis: micro (node-level), meso (community), macro (temporal)
- Publication-quality visualizations: PyVis, Plotly 2D/3D, Matplotlib, NetworkX
- Advanced analytics: t-SNE, UMAP, community detection, keyword burst detection
- Temporal evolution: concept timeline, growth rates, semantic drift
- Cross-domain bridge detection and network motif analysis
- Automated report generation with statistical validation
- Export: GraphML, GEXF, JSON, CSV, HTML, SVG, PNG (300 DPI), LaTeX tables

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn umap-learn

Run: streamlit run nanomaterials_concept_graph_v2.py

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
import copy
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from pathlib import Path

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (silhouette_score, r2_score, mean_absolute_error, 
                             mean_squared_error, adjusted_rand_score)
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.gridspec as gridspec
import seaborn as sns

from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="NanoGraph-Explorer: Core-Shell Ag-Cu Nanostructure Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PATHS & DIRECTORIES
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "exports")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ==============================================================================
# COLORMAP REGISTRY (50+)
# ==============================================================================
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

PUBLICATION_COLORMAPS = {
    "Nature Style": {"cmap": "viridis", "bg": "#ffffff", "grid": "#e8e8e8"},
    "Science Style": {"cmap": "plasma", "bg": "#ffffff", "grid": "#f0f0f0"},
    "Dark Presentation": {"cmap": "turbo", "bg": "#1a1a2e", "grid": "#2d2d44"},
    "ACS Nano Style": {"cmap": "cividis", "bg": "#ffffff", "grid": "#e8e8e8"},
    "High Contrast": {"cmap": "nipy_spectral", "bg": "#ffffff", "grid": "#d0d0d0"}
}


def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    """Convert matplotlib colormap to list of hex colors for Plotly/PyVis."""
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


# ==============================================================================
# ROBUST MULTI-FORMAT DATA LOADER
# ==============================================================================
def robust_load_file(filepath: Path):
    """Try multiple strategies to load a file (JSON/JSONL/CSV/XML/BIB)."""
    text = filepath.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"File is empty (0 bytes or only whitespace).")

    # Try JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Sanitize common JSON issues
    sanitized = re.sub(r'NaN', 'null', text)
    sanitized = re.sub(r'Infinity', 'null', sanitized)
    sanitized = re.sub(r'-Infinity', 'null', sanitized)
    sanitized = re.sub(r',(\s*[}\]])', r'\1', sanitized)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    # Try JSONL (one JSON per line)
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

    # Try CSV
    try:
        df = pd.read_csv(filepath)
        return df.to_dict(orient="records")
    except Exception:
        pass

    # Try TSV
    try:
        df = pd.read_csv(filepath, sep='\t')
        return df.to_dict(orient="records")
    except Exception:
        pass

    # Try BibTeX-like parsing (basic)
    bib_records = []
    entries = re.findall(r'@\w+\s*\{([^}]+)\}', text, re.DOTALL)
    if entries:
        for entry in entries:
            fields = re.findall(r'(\w+)\s*=\s*\{([^}]+)\}', entry)
            if fields:
                bib_records.append(dict(fields))
        if bib_records:
            return bib_records

    preview = text[:300]
    raise ValueError(f"Could not parse {filepath.name}. First 200 chars: {preview[:200]}...")


@st.cache_data(show_spinner=False)
def load_all_json_files(directory):
    """Load every supported file in directory and return list of (filepath, records)."""
    files = []
    for ext in ['*.json', '*.jsonl', '*.csv', '*.tsv', '*.bib']:
        files.extend(sorted(Path(directory).glob(ext)))
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
                st.code(f"Hex preview (first {len(raw_bytes)} bytes):\n{formatted}", language="text")
            except Exception:
                pass
    return loaded


@st.cache_data(show_spinner=False)
def build_master_dataframe(file_records):
    """Flatten all records into one DataFrame with rich metadata."""
    rows = []
    for fname, records in file_records:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec["_source_file"] = fname
            rec["_ ingestion_time"] = datetime.now().isoformat()
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    df = df.replace({float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA})

    # Auto-detect and normalize year columns
    year_cols = [c for c in df.columns if 'year' in c.lower() or 'date' in c.lower()]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Auto-detect DOI
    doi_cols = [c for c in df.columns if 'doi' in c.lower()]

    return df


# ==============================================================================
# CORE-SHELL AG-CU DOMAIN CONFIGURATION
# ==============================================================================
CORE_MATERIALS = [
    "cu@ag", "ag@cu", "cu ag core shell", "ag cu core shell", 
    "core shell copper silver", "core shell silver copper", 
    "cu@ag core shell", "ag@cu core shell",
    "copper silver core shell", "silver copper core shell",
    "cu ag nanoparticle", "ag cu nanoparticle", "copper silver nanoparticle",
    "cu ag bimetallic", "ag cu bimetallic", "copper silver bimetallic",
    "cu/ag core shell", "ag/cu core shell",
    "cu ag nanostructure", "ag cu nanostructure",
    "cu ag nanowire", "ag cu nanowire", "cu ag nanorod",
    "cu ag nanocrystal", "ag cu nanocrystal",
    "cu ag nanocomposite", "ag cu nanocomposite",
    "silver coated copper", "copper coated silver",
    "cu@ag core-shell", "ag@cu core-shell"
]

MATERIAL_PROPERTIES = [
    "lattice mismatch", "misfit strain", "interfacial strain", "coherency strain",
    "critical shell thickness", "epitaxial growth", "heteroepitaxy",
    "interfacial diffusion", "interdiffusion", "kirkendall effect", "void formation",
    "shell thickness", "core diameter", "core size", "shell volume",
    "surface plasmon resonance", "spr", "localized surface plasmon resonance", "lspr",
    "plasmon peak", "extinction coefficient", "absorption cross section",
    "catalytic activity", "electrocatalysis", "surface enhanced raman scattering", "sers",
    "enhancement factor", "hot spots", "electromagnetic enhancement",
    "electrical conductivity", "sheet resistance", "contact resistance", "resistivity",
    "thermal conductivity", "thermal stability", "oxidation resistance", "corrosion resistance",
    "antibacterial activity", "antimicrobial", "cytotoxicity", "biocompatibility",
    "work function", "band gap", "fermi level", "schottky barrier",
    "surface energy", "wettability", "hydrophobicity", "sers substrate",
    "refractive index", "dielectric function", "extinction spectrum"
]

SYNTHESIS_METHODS = [
    "seed mediated growth", "seed-mediated growth", "seed mediated",
    "galvanic replacement", "galvanic displacement", "transmetalation",
    "co-reduction", "coreduction", "simultaneous reduction",
    "successive reduction", "stepwise reduction", "sequential reduction",
    "chemical reduction", "thermal reduction", "polyol method", "polyol process",
    "solvothermal", "hydrothermal", "microemulsion", "reverse microemulsion",
    "electroless deposition", "electroless plating", "electrochemical deposition",
    "magnetron sputtering", "sputtering", "physical vapor deposition", "pvd",
    "chemical vapor deposition", "cvd", "atomic layer deposition", "ald",
    "molecular beam epitaxy", "mbe",
    "photochemical reduction", "photoreduction", "radiolytic reduction",
    "sonochemical", "microwave assisted", "laser ablation",
    "green synthesis", "biological synthesis", "plant extract",
    "capping agent", "surfactant", "pvp", "ctab", "oleylamine", "oleic acid",
    "ligand exchange", "surface functionalization", "self-assembly"
]

STRUCTURE_CHARACTERIZATION = [
    "transmission electron microscopy", "tem", "high resolution tem", "hrtem",
    "scanning transmission electron microscopy", "stem", "haadf-stem", "adf-stem",
    "energy dispersive x-ray spectroscopy", "eds", "edx", "stem-eds", "tem-eds",
    "elemental mapping", "line scan", "line profile", "eds mapping", "edx mapping",
    "electron energy loss spectroscopy", "eels", "stem-eels",
    "x-ray diffraction", "xrd", "xrd peak shift", "peak broadening", "scherrer equation",
    "selected area electron diffraction", "saed", "x-ray photoelectron spectroscopy", "xps",
    "uv-vis spectroscopy", "uv-vis-nir", "extinction spectrum", "absorption spectrum",
    "dynamic light scattering", "dls", "zeta potential", "hydrodynamic diameter",
    "small angle x-ray scattering", "saxs", "x-ray absorption spectroscopy", "xas",
    "extended x-ray absorption fine structure", "exafs", "xanes",
    "scanning electron microscopy", "sem", "fe-sem",
    "atomic force microscopy", "afm",
    "energy dispersive spectroscopy", "eds",
    "inductively coupled plasma", "icp", "icp-oes", "icp-ms",
    "core shell morphology", "yolk shell", "nanocavity", "porous shell",
    "lattice fringe", "interplanar spacing", "d-spacing", "crystal structure",
    "fcc", "face centered cubic", "epitaxial relationship", "orientation relationship"
]

COMPUTATIONAL_METHODS = [
    "density functional theory", "dft", "ab initio", "first principles",
    "molecular dynamics", "md", "classical md", "ab initio md", "aimd",
    "finite difference time domain", "fdtd", "discrete dipole approximation", "dda",
    "boundary element method", "bem", "finite element method", "fem",
    "mie theory", "mie scattering", "gans theory", "discrete dipole",
    "monte carlo", "kinetic monte carlo", "kmc", "metropolis algorithm",
    "machine learning potential", "ml potential", "neural network potential", "nnp",
    "molecular statics", "nudged elastic band", "neb",
    "comsol multiphysics", "lumerical", "meep",
    "phase field method", "pfm", "diffusion simulation", "finite element"
]

FUNCTIONAL_PROPERTIES = [
    "surface enhanced raman scattering", "sers", "sers substrate", "sers activity",
    "hot spot", "electromagnetic enhancement", "chemical enhancement",
    "localized surface plasmon resonance", "lspr", "plasmonic", "plasmon resonance",
    "refractive index sensitivity", "figure of merit", "fom",
    "catalytic activity", "electrocatalysis", "photocatalysis", "plasmonic catalysis",
    "surface catalysis", "co catalysis", "synergistic effect",
    "antibacterial", "antimicrobial", "bactericidal", "antibiofilm",
    "biocompatibility", "cytotoxicity", "cell viability", "drug delivery",
    "electrical conductivity", "conductivity", "sheet resistance", "interconnect",
    "conductive ink", "flexible electronics", "stretchable electronics", "wearable",
    "transparent conductor", "printed electronics", "solder", "die attach",
    "thermal conductivity", "thermal interface material", "tim",
    "photothermal therapy", "ptt", "photodynamic therapy", "pdt", "theranostics",
    "biosensor", "chemical sensor", "gas sensor", "colorimetric sensor",
    "optical filter", "absorber", "reflective", "structural color"
]

ALL_DOMAIN_KEYWORDS = (CORE_MATERIALS + MATERIAL_PROPERTIES + SYNTHESIS_METHODS +
                       STRUCTURE_CHARACTERIZATION + COMPUTATIONAL_METHODS + FUNCTIONAL_PROPERTIES)

NANOMATERIALS_PATTERNS = [
    r'\b(?:cu@ag|ag@cu|cu/ag|ag/cu)\b',
    r'\b(?:core\s*shell\s*(?:cu|ag|copper|silver|cu\s*ag|ag\s*cu|copper\s*silver|silver\s*copper))\b',
    r'\b(?:cu\s*ag|ag\s*cu|copper\s*silver|silver\s*copper)\s*(?:bimetallic|nanoparticle|nanostructure|nanocrystal|nanowire|nanorod|core\s*shell)\b',
    r'\b(?:bimetallic\s*(?:nanoparticle|nanowire|nanorod|nanostructure|core\s*shell))\b',
    r'\b(?:seed\s*mediated|galvanic\s*replacement|galvanic\s*displacement|co\s*reduction)\b',
    r'\b(?:lattice\s*mismatch|misfit\s*strain|interfacial\s*strain|epitaxial\s*growth)\b',
    r'\b(?:shell\s*thickness|core\s*diameter|core\s*size|critical\s*thickness)\b',
    r'\b(?:surface\s*plasmon|localized\s*surface\s*plasmon|lspr|sers|hot\s*spot)\b',
    r'\b(?:haadf\s*stem|stem\s*eds|tem\s*eds|elemental\s*mapping|line\s*profile)\b',
    r'\b(?:interdiffusion|kirkendall\s*effect|void\s*formation|interfacial\s*diffusion)\b',
    r'\b(?:pvp|ctab|oleylamine|oleic\s*acid|capping\s*agent|surfactant)\b',
    r'\b(?:fdtd|dda|mie\s*theory|comsol|lumerical)\b',
    r'\b(?:\d+(?:\.\d+)?\s*(?:nm|µm|micrometer|angstrom|å))\b',
    r'\b(?:uv\s*vis|extinction\s*spectrum|absorption\s*spectrum|plasmon\s*peak)\b'
]

NANOMATERIALS_CATEGORY_MAPPING = {
    r'cu@ag|ag@cu|cu/ag|ag/cu|copper\s*@\s*silver|silver\s*@\s*copper': 'core_shell_structure',
    r'core\s*shell\s*(?:cu|ag|copper|silver|bimetallic)': 'core_shell_structure',
    r'bimetallic\s*(?:cu|ag|copper|silver|nanoparticle|nanostructure)': 'bimetallic_system',
    r'seed\s*mediated|galvanic\s*replacement|galvanic\s*displacement|co\s*reduction': 'synthesis_method',
    r'polyol|chemical\s*reduction|thermal\s*reduction|microemulsion|electroless': 'synthesis_method',
    r'lattice\s*mismatch|misfit\s*strain|interfacial\s*strain|epitaxial|coherency': 'interfacial_structure',
    r'shell\s*thickness|core\s*diameter|core\s*size|critical\s*thickness|shell\s*volume': 'morphology_dimension',
    r'interdiffusion|kirkendall|void\s*formation|interfacial\s*diffusion|diffusion': 'interfacial_diffusion',
    r'surface\s*plasmon|lspr|sers|hot\s*spot|extinction\s*spectrum|plasmon\s*peak': 'plasmonic_optical',
    r'catalytic|electrocatalysis|photocatalysis|plasmonic\s*catalysis|synergistic': 'catalytic_activity',
    r'antibacterial|antimicrobial|bactericidal|cytotoxicity|biocompatibility': 'biomedical_property',
    r'electrical\s*conductivity|sheet\s*resistance|conductive\s*ink|interconnect': 'electronic_property',
    r'thermal\s*conductivity|thermal\s*stability|thermal\s*interface': 'thermal_property',
    r'fdtd|dda|mie\s*theory|comsol|lumerical|dft|molecular\s*dynamics': 'computational_method',
    r'haadf\s*stem|stem\s*eds|tem\s*eds|elemental\s*mapping|line\s*profile': 'advanced_characterization',
    r'uv\s*vis|xrd|xps|dls|zeta\s*potential|saxs|exafs': 'standard_characterization',
    r'pvp|ctab|oleylamine|oleic\s*acid|capping\s*agent|surfactant|ligand': 'surface_chemistry',
    r'biosensor|chemical\s*sensor|gas\s*sensor|colorimetric|photothermal': 'application_device'
}

CATEGORY_DISPLAY_NAMES = {
    'core_shell_structure': 'Core-Shell Structure',
    'bimetallic_system': 'Bimetallic System',
    'synthesis_method': 'Synthesis Method',
    'interfacial_structure': 'Interfacial Structure',
    'morphology_dimension': 'Morphology & Dimension',
    'interfacial_diffusion': 'Interfacial Diffusion',
    'plasmonic_optical': 'Plasmonic & Optical',
    'catalytic_activity': 'Catalytic Activity',
    'biomedical_property': 'Biomedical Property',
    'electronic_property': 'Electronic Property',
    'thermal_property': 'Thermal Property',
    'computational_method': 'Computational Method',
    'advanced_characterization': 'Advanced Characterization',
    'standard_characterization': 'Standard Characterization',
    'surface_chemistry': 'Surface Chemistry',
    'application_device': 'Application & Device',
    'general': 'General'
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    if num_abstracts <= 50:
        return {
            "MIN_CONCEPT_FREQ": 2, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 1, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.72, "COOCCURRENCE_WEIGHT": 0.5,
            "SEMANTIC_WEIGHT": 0.5, "CLUSTER_SIMILARITY": 0.75,
            "TOP_N_CONCEPTS": 200, "MAX_CONCEPT_LENGTH": 6
        }
    elif num_abstracts <= 500:
        return {
            "MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 2, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.78, "COOCCURRENCE_WEIGHT": 0.7,
            "SEMANTIC_WEIGHT": 0.3, "CLUSTER_SIMILARITY": 0.72,
            "TOP_N_CONCEPTS": 500, "MAX_CONCEPT_LENGTH": 8
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 5, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 3, "USE_SEMANTIC_CLUSTERING": False,
            "SIMILARITY_THRESHOLD": 0.85, "COOCCURRENCE_WEIGHT": 0.9,
            "SEMANTIC_WEIGHT": 0.1, "CLUSTER_SIMILARITY": 0.68,
            "TOP_N_CONCEPTS": 1000, "MAX_CONCEPT_LENGTH": 10
        }


# ==============================================================================
# DEVICE & MODEL MANAGEMENT
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


# ==============================================================================
# CONCEPT EXTRACTION & NORMALIZATION
# ==============================================================================
def is_valid_nanomaterials_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain = any(kw.lower() in concept_lower for kw in ALL_DOMAIN_KEYWORDS)
    has_pattern = any(re.search(p, concept, re.I) for p in NANOMATERIALS_PATTERNS)
    generic = {'study', 'analysis', 'effect', 'role', 'investigation', 'research',
               'method', 'approach', 'paper', 'work', 'using', 'based', 'novel',
               'new', 'recent', 'various', 'different', 'significant', 'important',
               'report', 'demonstrate', 'show', 'result', 'data', 'find'}
    has_generic = any(term in concept_lower.split() for term in generic)
    words = concept.split()
    if len(words) < 2 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic


def normalize_nanomaterials_term(concept: str) -> str:
    concept = concept.lower().strip()
    # Normalize core materials
    concept = re.sub(r'\bcu@ag\b', 'cu@ag core shell', concept)
    concept = re.sub(r'\bag@cu\b', 'ag@cu core shell', concept)
    concept = re.sub(r'\bcu/ag\b', 'cu@ag core shell', concept)
    concept = re.sub(r'\bag/cu\b', 'ag@cu core shell', concept)
    concept = re.sub(r'\bcore\s*shell\s*copper\s*silver\b', 'cu@ag core shell', concept)
    concept = re.sub(r'\bcore\s*shell\s*silver\s*copper\b', 'ag@cu core shell', concept)
    concept = re.sub(r'\bcopper\s*silver\s*nanoparticle\b', 'cu ag nanoparticle', concept)
    concept = re.sub(r'\bsilver\s*copper\s*nanoparticle\b', 'ag cu nanoparticle', concept)
    concept = re.sub(r'\bcopper\s*silver\s*bimetallic\b', 'cu ag bimetallic', concept)
    # Normalize synthesis
    concept = re.sub(r'\bseed\s*mediated\sgrowth\b', 'seed mediated growth', concept)
    concept = re.sub(r'\bgalvanic\s*replacement\b', 'galvanic replacement', concept)
    concept = re.sub(r'\bgalvanic\s*displacement\b', 'galvanic replacement', concept)
    concept = re.sub(r'\bco\s*reduction\b', 'co-reduction', concept)
    # Normalize characterization
    concept = re.sub(r'\bhigh\s*resolution\s*tem\b', 'hrtem', concept)
    concept = re.sub(r'\bscanning\s*transmission\s*electron\s*microscopy\b', 'stem', concept)
    concept = re.sub(r'\benergy\s*dispersive\s*x-ray\b', 'eds', concept)
    concept = re.sub(r'\belectron\s*energy\s*loss\s*spectroscopy\b', 'eels', concept)
    concept = re.sub(r'\bx-ray\s*diffraction\b', 'xrd', concept)
    concept = re.sub(r'\bx-ray\s*photoelectron\s*spectroscopy\b', 'xps', concept)
    concept = re.sub(r'\buv-vis\s*spectroscopy\b', 'uv-vis', concept)
    # Normalize computational
    concept = re.sub(r'\bdensity\s*functional\s*theory\b', 'dft', concept)
    concept = re.sub(r'\bab\s*initio\b', 'ab initio', concept)
    concept = re.sub(r'\bfirst\s*principles\b', 'first principles', concept)
    concept = re.sub(r'\bmolecular\s*dynamics\b', 'molecular dynamics', concept)
    concept = re.sub(r'\bfinite\s*element\b', 'finite element', concept)
    concept = re.sub(r'\bfinite\s*difference\s*time\s*domain\b', 'fdtd', concept)
    concept = re.sub(r'\bdiscrete\s*dipole\s*approximation\b', 'dda', concept)
    # Normalize units
    concept = re.sub(r'\bnm\b', 'nm', concept)
    concept = re.sub(r'\bµm\b', 'um', concept)
    return concept


def extract_concepts_from_text(text: str) -> List[str]:
    concepts = set()
    text_lower = text.lower()
    # Pattern-based extraction
    for pattern in NANOMATERIALS_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    # Noun phrase extraction for nanomaterials domain
    noun_pattern = r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,4}(?:nanoparticle|nanowire|nanorod|nanostructure|nanocrystal|nanotube|nanosheet|nanoplate|nanocube|nanosphere|nanocluster|nanocomposite|thin\s*film|coating|layer|interface|boundary|defect|dislocation|twin|precipitate|grain|phase|structure|morphology|property|performance|mechanism|process|method|technique|analysis|simulation|model|design|optimization)\b'
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_nanomaterials_concept(concept):
            concepts.add(concept)
    # Context-based extraction around domain keywords
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
    # Material-property pairs
    material_prop_pattern = r'\b([A-Z][a-z]+(?:\d+(?:\.\d+)?)?(?:[\s\-][A-Z][a-z]?\d*)+)\b\s+(?:with|having|exhibiting|showing|demonstrating|achieving|reaching|delivering|providing|offering)\s+(?:a\s+)?([\d\.]+\s*(?:nm|um|µm|angstrom|å|nm/riu|mv/dec|ma/cm2|s/cm|w/mk))\b'
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
        # Extract core-shell specific metrics
        size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:nm|um|µm)', combined_text, re.I)
        if size_matches: metrics['size_nm_um'] = [float(m) for m in size_matches]
        shell_matches = re.findall(r'shell\s*(?:thickness|width)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:nm|um|µm)', combined_text, re.I)
        if shell_matches: metrics['shell_thickness_nm'] = [float(m) for m in shell_matches]
        core_matches = re.findall(r'core\s*(?:diameter|size|radius)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:nm|um|µm)', combined_text, re.I)
        if core_matches: metrics['core_diameter_nm'] = [float(m) for m in core_matches]
        wavelength_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:nm)\s*(?:plasmon|peak|absorption|extinction|wavelength|spr|lspr)', combined_text, re.I)
        if wavelength_matches: metrics['plasmon_peak_nm'] = [float(m) for m in wavelength_matches]
        sensitivity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:nm/riu|nm/riU)', combined_text, re.I)
        if sensitivity_matches: metrics['refractive_index_sensitivity'] = [float(m) for m in sensitivity_matches]
        enhancement_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:x|×|\*)?\s*(?:enhancement\s*factor|ef|sers\s*ef)', combined_text, re.I)
        if enhancement_matches: metrics['enhancement_factor'] = [float(m) for m in enhancement_matches]

        all_metrics.append(metrics)
        concepts = extract_concepts_from_text(combined_text)
        normalized = [normalize_nanomaterials_term(c) for c in concepts]
        all_concepts.append(normalized)
    return all_concepts, all_metrics


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


def normalize_and_filter_concepts(all_concepts: List[List[str]], config: Dict) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
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
            if any(re.search(p, concept, re.I) for p in [r'\bcu@ag', r'\bag@cu', r'\bcore\s*shell']):
                concept_to_abstract[concept] = 'core_shell_structure'
            elif any(re.search(p, concept, re.I) for p in [r'\bbimetallic', r'\bcu\s*ag', r'\bag\s*cu']):
                concept_to_abstract[concept] = 'bimetallic_system'
            elif any(re.search(p, concept, re.I) for p in [r'\bseed\s*mediated', r'\bgalvanic', r'\bco-reduction']):
                concept_to_abstract[concept] = 'synthesis_method'
            elif any(re.search(p, concept, re.I) for p in [r'\blspr', r'\bsers', r'\bplasmon']):
                concept_to_abstract[concept] = 'plasmonic_optical'
            else:
                concept_to_abstract[concept] = 'general'
    return concept_to_abstract


# ==============================================================================
# CONCEPT DISTILLATION
# ==============================================================================
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


# ==============================================================================
# GRAPH CONSTRUCTION
# ==============================================================================
def build_hybrid_graph(all_concepts: List[List[str]], valid_concepts: List[str],
                        concept_to_id: Dict[str, int], embed_model=None, config: Dict = None) -> nx.Graph:
    if config is None:
        config = get_adaptive_config(3000)
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c, frequency=0)
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                    nx_graph[u][v]['cooccurrence'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1, cooccurrence=1, semantic=0, edge_type='cooccurrence')
                nx_graph.nodes[u]['frequency'] = nx_graph.nodes[u].get('frequency', 0) + 1
                nx_graph.nodes[v]['frequency'] = nx_graph.nodes[v].get('frequency', 0) + 1
    if embed_model and len(valid_concepts) >= 10:
        try:
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
                                         semantic=sim, edge_type='semantic')
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")
    cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.9)
    sem_weight = config.get("SEMANTIC_WEIGHT", 0.1)
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    return nx_graph


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


# ==============================================================================
# GNN MODEL
# ==============================================================================
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


# ==============================================================================
# RESEARCH DIRECTION SCORING
# ==============================================================================
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


# ==============================================================================
# MATHEMATICAL VALIDATION
# ==============================================================================
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

# ==============================================================================
# ADVANCED FEATURE 1: KEYWORD BURST DETECTION (Kleinberg Algorithm-inspired)
# ==============================================================================
def detect_keyword_bursts(df_filtered, concept_abstract_map, valid_concepts, year_col='Year', 
                           gamma: float = 1.0, sigma: float = 1.0):
    """
    Detect keyword bursts using a simplified Kleinberg algorithm.
    Returns DataFrame with burst levels for each concept.
    """
    if year_col not in df_filtered.columns:
        return pd.DataFrame()

    df_years = df_filtered[df_filtered[year_col].notna()].copy()
    if df_years.empty:
        return pd.DataFrame()

    df_years[year_col] = pd.to_numeric(df_years[year_col], errors="coerce")
    df_years = df_years.dropna(subset=[year_col])
    if df_years.empty:
        return pd.DataFrame()

    years = sorted(df_years[year_col].unique())
    if len(years) < 3:
        return pd.DataFrame()

    abs_year = {i: y for i, y in df_years[year_col].items() if i < len(df_filtered)}

    burst_data = []
    for concept in valid_concepts:
        abs_list = concept_abstract_map.get(concept, [])
        concept_years = [abs_year[idx] for idx in abs_list if idx in abs_year]
        if not concept_years:
            continue

        year_counts = Counter(concept_years)
        total = len(concept_years)

        # Compute burst score: ratio of max consecutive years to total span
        if total < 2:
            continue

        year_list = sorted(concept_years)
        span = max(year_list) - min(year_list) + 1
        if span < 2:
            continue

        # Find longest consecutive run
        max_run = 1
        current_run = 1
        for i in range(1, len(year_list)):
            if year_list[i] == year_list[i-1] + 1:
                current_run += 1
                max_run = max(max_run, current_run)
            elif year_list[i] != year_list[i-1]:
                current_run = 1

        # Burst score: concentration + recency
        concentration = max_run / span if span > 0 else 0
        recency = year_list.count(max(years)) / total if total > 0 else 0
        burst_score = concentration * (1 + gamma * recency)

        burst_data.append({
            'concept': concept,
            'burst_score': burst_score,
            'concentration': concentration,
            'recency': recency,
            'total_occurrences': total,
            'year_span': span,
            'first_year': min(year_list),
            'last_year': max(year_list),
            'max_consecutive': max_run
        })

    return pd.DataFrame(burst_data).sort_values('burst_score', ascending=False)


# ==============================================================================
# ADVANCED FEATURE 2: CROSS-DOMAIN BRIDGE DETECTION
# ==============================================================================
def detect_cross_domain_bridges(nx_graph, valid_concepts):
    """
    Detect concepts that bridge between different knowledge domains.
    A bridge concept connects nodes from different categories with high betweenness.
    """
    if nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()

    category_map = abstract_concepts_to_categories(valid_concepts)

    # Compute betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(nx_graph, normalized=True, k=min(100, nx_graph.number_of_nodes()))
    except Exception:
        betweenness = {n: 0 for n in nx_graph.nodes()}

    bridge_scores = []
    for node in nx_graph.nodes():
        neighbors = list(nx_graph.neighbors(node))
        if len(neighbors) < 2:
            continue

        neighbor_categories = set()
        for nb in neighbors:
            neighbor_categories.add(category_map.get(nb, 'general'))

        # Bridge score: betweenness * category diversity
        cat_diversity = len(neighbor_categories) / len(neighbors) if neighbors else 0
        bridge_score = betweenness.get(node, 0) * cat_diversity

        bridge_scores.append({
            'concept': node,
            'bridge_score': bridge_score,
            'betweenness': betweenness.get(node, 0),
            'category_diversity': cat_diversity,
            'neighbor_count': len(neighbors),
            'categories_spanned': list(neighbor_categories),
            'own_category': category_map.get(node, 'general')
        })

    return pd.DataFrame(bridge_scores).sort_values('bridge_score', ascending=False)


# ==============================================================================
# ADVANCED FEATURE 3: NETWORK MOTIF ANALYSIS
# ==============================================================================
def analyze_network_motifs(nx_graph, valid_concepts, max_size: int = 4):
    """
    Analyze common network motifs (triangles, stars, chains) in the concept graph.
    """
    if nx_graph.number_of_nodes() < 3:
        return {}

    motifs = {
        'triangles': 0,
        'stars': 0,
        'chains': 0,
        'cliques_4': 0,
        'total_nodes': nx_graph.number_of_nodes(),
        'total_edges': nx_graph.number_of_edges()
    }

    # Count triangles
    try:
        triangles = nx.triangles(nx_graph)
        motifs['triangles'] = sum(triangles.values()) // 3
        motifs['avg_clustering'] = nx.average_clustering(nx_graph)
    except Exception:
        pass

    # Count star motifs (nodes with degree >= 3)
    degree_sequence = [d for n, d in nx_graph.degree()]
    motifs['stars'] = sum(1 for d in degree_sequence if d >= 3)
    motifs['max_degree'] = max(degree_sequence) if degree_sequence else 0
    motifs['avg_degree'] = np.mean(degree_sequence) if degree_sequence else 0

    # Count 4-cliques
    try:
        cliques = list(nx.find_cliques(nx_graph))
        motifs['cliques_4'] = sum(1 for c in cliques if len(c) >= 4)
    except Exception:
        pass

    # Transitivity
    try:
        motifs['transitivity'] = nx.transitivity(nx_graph)
    except Exception:
        motifs['transitivity'] = 0.0

    return motifs


# ==============================================================================
# ADVANCED FEATURE 4: SEMANTIC DRIFT DETECTION
# ==============================================================================
def detect_semantic_drift(valid_concepts, concept_abstract_map, all_texts, embed_model, 
                           year_col='Year', df_filtered=None):
    """
    Detect how concept meanings shift over time by comparing early vs late embeddings.
    """
    if df_filtered is None or year_col not in df_filtered.columns:
        return pd.DataFrame()

    df_years = df_filtered[df_filtered[year_col].notna()].copy()
    df_years[year_col] = pd.to_numeric(df_years[year_col], errors="coerce")
    df_years = df_years.dropna(subset=[year_col])
    if df_years.empty or len(df_years) < 10:
        return pd.DataFrame()

    median_year = df_years[year_col].median()
    abs_year = {i: y for i, y in df_years[year_col].items() if i < len(df_filtered)}

    drift_data = []
    for concept in valid_concepts:
        abs_list = concept_abstract_map.get(concept, [])
        early_texts = [all_texts[i] for i in abs_list if i in abs_year and abs_year[i] <= median_year]
        late_texts = [all_texts[i] for i in abs_list if i in abs_year and abs_year[i] > median_year]

        if len(early_texts) < 2 or len(late_texts) < 2:
            continue

        try:
            early_emb = embed_model.encode(early_texts, show_progress_bar=False, batch_size=32)
            late_emb = embed_model.encode(late_texts, show_progress_bar=False, batch_size=32)

            early_centroid = np.mean(early_emb, axis=0)
            late_centroid = np.mean(late_emb, axis=0)

            drift = 1 - cosine_similarity([early_centroid], [late_centroid])[0, 0]

            drift_data.append({
                'concept': concept,
                'semantic_drift': float(drift),
                'early_count': len(early_texts),
                'late_count': len(late_texts),
                'median_year': median_year
            })
        except Exception:
            continue

    return pd.DataFrame(drift_data).sort_values('semantic_drift', ascending=False)


# ==============================================================================
# ADVANCED FEATURE 5: CONCEPT GENEALOGY (ANCESTRY TRACKING)
# ==============================================================================
def build_concept_genealogy(nx_graph, valid_concepts, concept_abstract_map):
    """
    Build a concept genealogy by tracing shortest paths and identifying parent-child
    relationships based on frequency and connectivity patterns.
    """
    if nx_graph.number_of_nodes() < 3:
        return pd.DataFrame()

    genealogy = []
    freq_map = {c: len(concept_abstract_map.get(c, [])) for c in valid_concepts}

    for concept in valid_concepts:
        if concept not in nx_graph:
            continue

        # Find "parent" concepts: higher frequency, directly connected
        parents = []
        for neighbor in nx_graph.neighbors(concept):
            if freq_map.get(neighbor, 0) > freq_map.get(concept, 0):
                parents.append(neighbor)

        # Find "children" concepts: lower frequency, directly connected
        children = []
        for neighbor in nx_graph.neighbors(concept):
            if freq_map.get(neighbor, 0) < freq_map.get(concept, 0):
                children.append(neighbor)

        # Find "siblings": same frequency tier, shared parents
        siblings = []
        for other in valid_concepts:
            if other != concept and other in nx_graph:
                shared = set(nx_graph.neighbors(concept)) & set(nx_graph.neighbors(other))
                if len(shared) > 0 and abs(freq_map.get(other, 0) - freq_map.get(concept, 0)) < 3:
                    siblings.append(other)

        genealogy.append({
            'concept': concept,
            'frequency': freq_map.get(concept, 0),
            'parents': parents,
            'n_parents': len(parents),
            'children': children,
            'n_children': len(children),
            'siblings': siblings,
            'n_siblings': len(siblings),
            'generation': 0  # Will be computed
        })

    df_genealogy = pd.DataFrame(genealogy)

    # Compute generations (topological layers)
    if not df_genealogy.empty:
        # Root concepts: no parents
        roots = df_genealogy[df_genealogy['n_parents'] == 0]['concept'].tolist()
        generation_map = {c: 0 for c in roots}

        # BFS to assign generations
        queue = deque(roots)
        visited = set(roots)
        while queue:
            current = queue.popleft()
            current_gen = generation_map.get(current, 0)
            # Find children
            row = df_genealogy[df_genealogy['concept'] == current]
            if not row.empty:
                children = row.iloc[0]['children']
                for child in children:
                    if child not in visited:
                        generation_map[child] = current_gen + 1
                        visited.add(child)
                        queue.append(child)

        df_genealogy['generation'] = df_genealogy['concept'].map(generation_map).fillna(0).astype(int)

    return df_genealogy


# ==============================================================================
# ADVANCED FEATURE 6: AUTOMATED REPORT GENERATION
# ==============================================================================
def generate_analysis_report(data, metrics, val_metrics, top_scores, distill_df, 
                              burst_df, bridge_df, motif_data, drift_df, genealogy_df):
    """Generate a comprehensive Markdown report of the analysis."""
    report = []
    report.append("# NanoGraph-Explorer Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## 1. Dataset Overview")
    report.append(f"- **Total Concepts:** {len(data['valid_concepts'])}")
    report.append(f"- **Total Edges:** {data['nx_graph'].number_of_edges()}")
    report.append(f"- **Documents Analyzed:** {len(data['all_texts'])}")
    report.append("")

    report.append("## 2. Graph Topology Metrics")
    report.append(f"- **Density:** {metrics.get('density', 0):.4f}")
    report.append(f"- **Average Degree:** {metrics.get('avg_degree', 0):.2f}")
    report.append(f"- **Clustering Coefficient:** {metrics.get('clustering', 0):.4f}")
    report.append(f"- **Connected Components:** {metrics.get('connected_components', 0)}")
    report.append(f"- **Modularity:** {val_metrics.get('modularity', 0):.4f}")
    report.append(f"- **Silhouette Score:** {val_metrics.get('silhouette_score', 0):.4f}")
    report.append("")

    report.append("## 3. Top Concepts by Distillation Efficiency")
    if not distill_df.empty:
        for _, row in distill_df.head(10).iterrows():
            report.append(f"- **{row['concept']}**: {row['distillation_efficiency']:.3f} (freq={row['frequency']})")
    report.append("")

    report.append("## 4. Top Research Direction Recommendations")
    if not top_scores.empty:
        for _, row in top_scores.head(10).iterrows():
            report.append(f"- **{row['concept_u']}** ↔ **{row['concept_v']}**: {row['composite_score']:.3f}")
    report.append("")

    report.append("## 5. Keyword Burst Detection")
    if not burst_df.empty:
        for _, row in burst_df.head(10).iterrows():
            report.append(f"- **{row['concept']}**: burst={row['burst_score']:.3f}, span={row['year_span']}y")
    report.append("")

    report.append("## 6. Cross-Domain Bridges")
    if not bridge_df.empty:
        for _, row in bridge_df.head(10).iterrows():
            report.append(f"- **{row['concept']}**: bridge={row['bridge_score']:.3f}, categories={row['category_diversity']:.2f}")
    report.append("")

    report.append("## 7. Network Motifs")
    report.append(f"- **Triangles:** {motif_data.get('triangles', 0)}")
    report.append(f"- **Star Nodes:** {motif_data.get('stars', 0)}")
    report.append(f"- **4-Cliques:** {motif_data.get('cliques_4', 0)}")
    report.append(f"- **Transitivity:** {motif_data.get('transitivity', 0):.4f}")
    report.append("")

    report.append("## 8. Semantic Drift")
    if not drift_df.empty:
        for _, row in drift_df.head(10).iterrows():
            report.append(f"- **{row['concept']}**: drift={row['semantic_drift']:.4f}")
    report.append("")

    return "\n".join(report)


# ==============================================================================
# ADVANCED FEATURE 7: PUBLICATION-QUALITY FIGURE EXPORT
# ==============================================================================
def export_publication_figure(nx_graph, concept_abstract_map, valid_concepts, 
                             figure_type: str = "network", dpi: int = 300,
                             cmap_name: str = "viridis", width: int = 12, height: int = 10):
    """Generate publication-quality figures using Matplotlib."""
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    if figure_type == "network":
        pos = nx.spring_layout(nx_graph, seed=42, k=2.5)
        node_colors = [get_nanomaterials_category_color(n) for n in nx_graph.nodes()]
        node_sizes = [max(100, min(800, len(concept_abstract_map.get(n, [])) * 30)) for n in nx_graph.nodes()]

        nx.draw_networkx_edges(nx_graph, pos, alpha=0.3, edge_color='gray', width=0.5)
        nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=node_sizes, 
                               alpha=0.9, edgecolors='white', linewidths=1)
        nx.draw_networkx_labels(nx_graph, pos, font_size=6, font_weight='bold')
        plt.title("Core-Shell Ag-Cu Nanostructure Concept Network", fontsize=14, fontweight='bold')
        plt.axis('off')

    elif figure_type == "degree_distribution":
        degrees = [d for n, d in nx_graph.degree()]
        plt.hist(degrees, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel("Degree", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Degree Distribution", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

    elif figure_type == "community":
        try:
            from networkx.algorithms import community
            comms = list(community.greedy_modularity_communities(nx_graph))
            colors = get_colormap_colors(cmap_name, len(comms))
            node_color_map = {}
            for i, comm in enumerate(comms):
                for node in comm:
                    node_color_map[node] = colors[i % len(colors)]
            pos = nx.spring_layout(nx_graph, seed=42)
            node_colors = [node_color_map.get(n, '#999999') for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, node_color=node_colors, with_labels=True, 
                    node_size=300, font_size=6, edge_color='gray', alpha=0.6)
            plt.title("Community Structure", fontsize=14, fontweight='bold')
            plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f"Community detection failed: {e}", ha='center', va='center')
            plt.axis('off')

    # Save to bytes
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf.getvalue()

# ==============================================================================
# GRAPH EDITING WITH UNDO/REDO
# ==============================================================================
class GraphEditHistory:
    """Manage graph editing history with undo/redo capability."""
    def __init__(self, max_history: int = 50):
        self.history = deque(maxlen=max_history)
        self.redo_stack = deque(maxlen=max_history)
        self.current_index = -1

    def push_state(self, state):
        """Push a new state onto history."""
        # Remove any future states if we're not at the end
        while len(self.history) > self.current_index + 1:
            self.history.pop()
        self.history.append(copy.deepcopy(state))
        self.current_index = len(self.history) - 1
        self.redo_stack.clear()

    def undo(self):
        """Undo last edit."""
        if self.current_index > 0:
            self.redo_stack.append(copy.deepcopy(self.history[self.current_index]))
            self.current_index -= 1
            return copy.deepcopy(self.history[self.current_index])
        return None

    def redo(self):
        """Redo last undone edit."""
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.current_index += 1
            if self.current_index < len(self.history):
                self.history[self.current_index] = state
            else:
                self.history.append(state)
            return copy.deepcopy(state)
        return None

    def can_undo(self):
        return self.current_index > 0

    def can_redo(self):
        return len(self.redo_stack) > 0


def apply_graph_edits(nx_graph, concept_abstract_map, valid_concepts, edits):
    """
    Apply user edits to the graph.
    edits dict keys: 'remove_nodes', 'merge_nodes', 'rename_nodes', 'add_edges', 'filter_by_degree', 'filter_by_freq'
    Returns updated graph, concept_abstract_map, valid_concepts.
    """
    G = nx_graph.copy()
    map_abs = {k: list(v) for k, v in concept_abstract_map.items()}
    concepts = list(valid_concepts)

    # 1. Remove nodes
    for node in edits.get('remove_nodes', []):
        if node in G:
            G.remove_node(node)
            map_abs.pop(node, None)
            if node in concepts:
                concepts.remove(node)

    # 2. Merge nodes
    for merge_list, new_name in edits.get('merge_nodes', []):
        if len(merge_list) < 2 or not all(n in G for n in merge_list):
            continue
        combined_abstracts = []
        for n in merge_list:
            combined_abstracts.extend(map_abs.get(n, []))
        combined_abstracts = list(set(combined_abstracts))

        G.add_node(new_name, frequency=len(combined_abstracts))
        map_abs[new_name] = combined_abstracts

        neighbors = set()
        for n in merge_list:
            neighbors.update(G.neighbors(n))
        for nb in neighbors:
            if nb not in merge_list:
                w = sum(G[n][nb].get('weight', 1) for n in merge_list if G.has_edge(n, nb))
                G.add_edge(new_name, nb, weight=w, cooccurrence=w, edge_type='cooccurrence')

        for n in merge_list:
            G.remove_node(n)
            map_abs.pop(n, None)
            if n in concepts:
                concepts.remove(n)
        if new_name not in concepts:
            concepts.append(new_name)

    # 3. Rename nodes
    for old, new in edits.get('rename_nodes', []):
        if old in G and new not in G:
            adj = dict(G[old])
            attrs = dict(G.nodes[old])
            G.add_node(new, **attrs)
            for nb, data in adj.items():
                G.add_edge(new, nb, **data)
            G.remove_node(old)
            if old in map_abs:
                map_abs[new] = map_abs.pop(old)
            if old in concepts:
                idx = concepts.index(old)
                concepts[idx] = new

    # 4. Add edges
    for u, v, weight in edits.get('add_edges', []):
        if u in G and v in G and not G.has_edge(u, v):
            G.add_edge(u, v, weight=weight, cooccurrence=weight, edge_type='user_added')

    # 5. Filter by degree
    min_degree = edits.get('filter_by_degree', 0)
    if min_degree > 0:
        nodes_to_remove = [n for n in G.nodes() if G.degree(n) < min_degree]
        for node in nodes_to_remove:
            G.remove_node(node)
            map_abs.pop(node, None)
            if node in concepts:
                concepts.remove(node)

    # 6. Filter by frequency
    min_freq = edits.get('filter_by_freq', 0)
    if min_freq > 0:
        nodes_to_remove = [n for n in G.nodes() if len(map_abs.get(n, [])) < min_freq]
        for node in nodes_to_remove:
            if node in G:
                G.remove_node(node)
            map_abs.pop(node, None)
            if node in concepts:
                concepts.remove(node)

    return G, map_abs, concepts


# ==============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ==============================================================================
def get_nanomaterials_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    if any(c in concept_lower for c in ['cu@ag', 'ag@cu', 'cu/ag', 'ag/cu', 'core shell']):
        return "#1976D2"
    elif any(c in concept_lower for c in ['bimetallic', 'cu ag', 'ag cu', 'copper silver']):
        return "#0D47A1"
    elif any(c in concept_lower for c in ['lattice mismatch', 'misfit strain', 'interfacial strain', 'epitaxial', 'coherency']):
        return "#E91E63"
    elif any(c in concept_lower for c in ['shell thickness', 'core diameter', 'core size', 'morphology', 'dimension']):
        return "#9C27B0"
    elif any(c in concept_lower for c in ['interdiffusion', 'kirkendall', 'void formation', 'diffusion']):
        return "#FF9800"
    elif any(c in concept_lower for c in ['surface plasmon', 'lspr', 'sers', 'hot spot', 'extinction', 'plasmon peak']):
        return "#F44336"
    elif any(c in concept_lower for c in ['uv vis', 'absorption spectrum', 'refractive index']):
        return "#FF5722"
    elif any(c in concept_lower for c in ['seed mediated', 'galvanic', 'co-reduction', 'polyol', 'chemical reduction']):
        return "#00BCD4"
    elif any(c in concept_lower for c in ['pvp', 'ctab', 'oleylamine', 'oleic acid', 'capping agent', 'surfactant', 'ligand']):
        return "#009688"
    elif any(c in concept_lower for c in ['haadf', 'stem eds', 'tem eds', 'elemental mapping', 'line profile', 'eels']):
        return "#3F51B5"
    elif any(c in concept_lower for c in ['xrd', 'xps', 'dls', 'zeta', 'saxs', 'exafs', 'uv vis']):
        return "#795548"
    elif any(c in concept_lower for c in ['fdtd', 'dda', 'mie theory', 'comsol', 'lumerical', 'dft', 'molecular dynamics']):
        return "#4CAF50"
    elif any(c in concept_lower for c in ['catalytic', 'electrocatalysis', 'photocatalysis', 'synergistic']):
        return "#8BC34A"
    elif any(c in concept_lower for c in ['antibacterial', 'antimicrobial', 'bactericidal', 'cytotoxicity', 'biocompatibility']):
        return "#8E24AA"
    elif any(c in concept_lower for c in ['electrical conductivity', 'sheet resistance', 'conductive ink', 'interconnect']):
        return "#FFC107"
    elif any(c in concept_lower for c in ['thermal conductivity', 'thermal stability', 'thermal interface']):
        return "#FFEB3B"
    elif any(c in concept_lower for c in ['biosensor', 'chemical sensor', 'gas sensor', 'photothermal']):
        return "#00E676"
    else:
        return "#9E9E9E"


def render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=True,
                        min_node_size=8, max_node_size=40, cmap_name="viridis",
                        custom_labels=None, node_label_size=12, top_n_nodes=0,
                        theme=None, physics_preset=None):
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if physics_preset is None:
        physics_preset = PHYSICS_PRESETS["Stable (Default)"]

    pos = {}
    if len(nx_graph.nodes()) > 0:
        try:
            if len(nx_graph.nodes()) < 300:
                pos = nx.kamada_kawai_layout(nx_graph, weight='weight')
            else:
                pos = nx.spring_layout(nx_graph, k=2.5, iterations=200, seed=42, weight='weight')
        except Exception:
            pos = nx.spring_layout(nx_graph, k=2.5, iterations=200, seed=42, weight='weight')

    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))

    net = Network(
        height="780px", width="100%", bgcolor=theme['bg'], font_color=theme['font'],
        select_menu=True, notebook=False, cdn_resources='remote'
    )

    if physics_enabled and physics_preset.get("gravity", 0) != 0:
        net.set_options(f"""
        var options = {{
          "physics": {{
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {{
              "gravitationalConstant": {physics_preset['gravity']},
              "centralGravity": {physics_preset['central_gravity']},
              "springLength": {physics_preset['spring_length']},
              "springConstant": {physics_preset['spring_strength']},
              "damping": {physics_preset['damping']},
              "overlap": 0.15
            }},
            "stabilization": {{
              "enabled": true,
              "iterations": {physics_preset['stabilization']},
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

    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(min_node_size + freq * 1.2, min_node_size, max_node_size))
        color = get_nanomaterials_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        label = custom_labels.get(node, node) if custom_labels else node

        x, y = (pos.get(node, (0, 0))[0] * 1200, pos.get(node, (0, 0))[1] * 1200)

        net.add_node(
            node,
            label=label,
            size=size,
            x=x,
            y=y,
            color={
                'background': color,
                'border': theme['node_border'],
                'highlight': {'background': theme['highlight_bg'], 'border': '#ffffff'},
                'hover': {'background': theme['hover_bg'], 'border': '#ffffff'}
            },
            font={
                'color': theme['font'],
                'size': node_label_size,
                'face': 'Inter, Segoe UI, Roboto, sans-serif',
                'strokeWidth': 0,
                'vadjust': -6
            },
            title=(
                f"<div style='font-family:Inter,sans-serif;'>"
                f"<b style='font-size:14px;color:{theme['highlight_bg']};'>{node}</b><br>"
                f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>Degree:</span> {degree}<br>"
                f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>Frequency:</span> {freq}"
                f"</div>"
            ),
            borderWidth=2,
            borderWidthSelected=3,
            shadow={
                'enabled': True,
                'color': theme['shadow_color'],
                'size': 12,
                'x': 4,
                'y': 4
            },
            shape='dot',
            mass=max(1, 1 + freq * 0.05)
        )

    color_map = {
        'cooccurrence': theme['edge_cooccurrence'],
        'semantic':     theme['edge_semantic'],
        'bridge':       theme['edge_bridge'],
        'user_added':   '#FF00FF',
        'unknown':      theme['edge_unknown']
    }

    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        color = color_map.get(edge_type, color_map['unknown'])
        width = float(np.clip(w * 0.4, 0.8, 3.5))

        net.add_edge(
            u, v,
            value=float(np.clip(w, 0.5, 5)),
            width=width,
            color={
                'color': color,
                'highlight': theme['highlight_bg'],
                'hover': theme['hover_bg'],
                'opacity': 0.85
            },
            smooth={'type': 'continuous', 'roundness': 0.35},
            title=f"<span style='font-family:Inter,sans-serif;'>Weight: <b>{w:.2f}</b><br>Type: {edge_type}</span>"
        )

    html_content = net.generate_html()

    custom_css = f"""
    <style>
        body {{
            background: {theme['bg']};
            margin: 0;
            padding: 0;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        #mynetwork {{
            border-radius: 16px;
            box-shadow: 0 12px 48px {theme['shadow_color']};
            outline: none;
        }}
        div.vis-tooltip {{
            background: {theme['tooltip_bg']} !important;
            color: {theme['tooltip_text']} !important;
            border: 1px solid {theme['tooltip_border']} !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
            box-shadow: 0 8px 32px {theme['shadow_color']} !important;
            max-width: 320px !important;
            white-space: normal !important;
        }}
        div.vis-network div.vis-manipulation {{
            background: {theme['tooltip_bg']} !important;
            border-top: 1px solid {theme['tooltip_border']} !important;
            color: {theme['font']} !important;
        }}
    </style>
    """
    html_content = html_content.replace('</head>', custom_css + '</head>')

    st.components.v1.html(html_content, height=790, scrolling=True)

    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button("📥 Download Interactive Graph (HTML)", data=html_bytes,
                          file_name="core_shell_agcu_concept_graph.html", mime="text/html")
        del html_content, html_bytes
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
        edge_hover.extend([f"<b>{u} ↔ {v}</b><br>Weight: {w:.2f}<br>Type: {edge_type}"] * 2 + [None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=1, color=theme['edge_unknown']),
                            hoverinfo='text', hovertext=edge_hover, name='Connections')
    node_x, node_y, node_text, node_size, node_color, node_labels = [], [], [], [], [], []
    for i, node in enumerate(nx_graph.nodes()):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}")
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
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}")
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
    if len(nx_graph.edges()) > 0:
        edge_list = [(u, v, nx_graph[u][v].get('weight', 1)) for u, v in nx_graph.edges()]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        st.markdown("**🔗 Top 20 Strongest Connections:**")
        for i, (u, v, w) in enumerate(edge_list[:20], 1):
            edge_type = nx_graph[u][v].get('edge_type', 'unknown')
            st.markdown(f"{i}. `{u}` ↔ `{v}` (weight: {w:.2f}, type: {edge_type})")
    if len(concept_abstract_map) > 0:
        freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in nx_graph.nodes()]
        freq_data.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**📈 Top Concepts by Frequency:**")
        st.dataframe(pd.DataFrame(freq_data[:15], columns=["Concept", "Abstract Count"]), use_container_width=True)

# ==============================================================================
# NEW INNOVATIVE VISUALIZATIONS
# ==============================================================================

def render_timeline(df_filtered, concept_abstract_map, valid_concepts, year_col='Year'):
    """Display concept frequency over time as interactive line chart."""
    if year_col not in df_filtered.columns:
        st.info("No 'Year' column found. Timeline unavailable.")
        return
    df_years = df_filtered[df_filtered[year_col].notna()].copy()
    if df_years.empty:
        st.info("No valid years in data.")
        return
    df_years[year_col] = pd.to_numeric(df_years[year_col], errors="coerce")
    df_years = df_years.dropna(subset=[year_col])
    if df_years.empty:
        return

    abs_year = {i: y for i, y in df_years[year_col].items() if i < len(df_filtered)}
    years = sorted(set(abs_year.values()))
    if len(years) < 2:
        st.info("Need at least 2 distinct years for timeline.")
        return

    concept_year_counts = defaultdict(lambda: defaultdict(int))
    for concept, abs_list in concept_abstract_map.items():
        if concept not in valid_concepts:
            continue
        for idx in abs_list:
            if idx in abs_year:
                concept_year_counts[concept][abs_year[idx]] += 1

    data = []
    for concept, year_dict in concept_year_counts.items():
        for y in years:
            data.append({'concept': concept, 'year': y, 'count': year_dict.get(y, 0)})
    df_timeline = pd.DataFrame(data)
    if df_timeline.empty:
        st.info("No timeline data.")
        return

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
    freq = {c: len(concept_abstract_map.get(c, [])) for c in valid_concepts}
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_concepts = [c for c, _ in top]

    matrix = np.zeros((len(top_concepts), len(top_concepts)))
    for i, u in enumerate(top_concepts):
        for j, v in enumerate(top_concepts):
            if i == j:
                matrix[i,j] = 0
            else:
                u_abs = set(concept_abstract_map.get(u, []))
                v_abs = set(concept_abstract_map.get(v, []))
                matrix[i,j] = len(u_abs & v_abs)

    fig = px.imshow(matrix, x=top_concepts, y=top_concepts,
                    title='Co-occurrence Heatmap (Top Concepts)',
                    color_continuous_scale='Blues',
                    labels=dict(x='Concept', y='Concept', color='Co-occurrence'))
    fig.update_layout(xaxis=dict(tickangle=45), height=700)
    st.plotly_chart(fig, use_container_width=True)


def render_tsne_projection(valid_concepts, embed_model, concept_abstract_map, cmap_name='viridis'):
    """t-SNE projection of concept embeddings with category coloring."""
    if len(valid_concepts) < 5:
        st.info("Need at least 5 concepts for t-SNE.")
        return
    st.info("Computing t-SNE (may take a few seconds)...")
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        tsne = TSNE(n_components=2, perplexity=min(30, len(valid_concepts)-1), random_state=42)
        coords = tsne.fit_transform(embeddings)
        categories = [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]
        freqs = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
        df_tsne = pd.DataFrame({
            'concept': valid_concepts, 'x': coords[:,0], 'y': coords[:,1],
            'category': categories, 'frequency': freqs
        })
        unique_cats = df_tsne['category'].unique()
        fig = px.scatter(df_tsne, x='x', y='y', color='category', size='frequency',
                         hover_name='concept',
                         title='t-SNE Projection of Concept Embeddings',
                         labels={'x':'t-SNE 1', 'y':'t-SNE 2'},
                         size_max=20,
                         color_discrete_sequence=get_colormap_colors(cmap_name, len(unique_cats)))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"t-SNE failed: {e}")


def render_community_detection(nx_graph, valid_concepts):
    """Detect communities and visualize with distinct colors."""
    if nx_graph.number_of_nodes() < 3:
        st.info("Graph too small for community detection.")
        return
    try:
        from networkx.algorithms import community
        comms = list(community.greedy_modularity_communities(nx_graph))
        comm_map = {}
        for i, comm in enumerate(comms):
            for node in comm:
                comm_map[node] = i
        G = nx_graph.copy()
        colors = get_colormap_colors('tab20', len(comms))
        node_colors = [colors[comm_map.get(n, 0) % len(colors)] for n in G.nodes()]
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(12,10))
        nx.draw(G, pos, node_color=node_colors, with_labels=True, ax=ax,
                node_size=400, font_size=8, edge_color='gray', alpha=0.7)
        ax.set_title('Community Structure (Modularity)', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Community detection failed: {e}")


def render_concept_growth(df_filtered, concept_abstract_map, valid_concepts, year_col='Year'):
    """Compute and visualize concept growth rates over time."""
    if year_col not in df_filtered.columns:
        st.info("No Year column. Growth analysis unavailable.")
        return
    df_years = df_filtered[df_filtered[year_col].notna()].copy()
    if df_years.empty:
        return
    df_years[year_col] = pd.to_numeric(df_years[year_col], errors="coerce")
    df_years = df_years.dropna(subset=[year_col])
    if df_years.empty:
        return

    min_year = df_years[year_col].min()
    max_year = df_years[year_col].max()
    if max_year - min_year < 1:
        st.info("Insufficient year range.")
        return

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
        year_counts = Counter(years)
        early_count = sum(year_counts.get(y, 0) for y in range(int(min_year), int(min_year)+2))
        late_count = sum(year_counts.get(y, 0) for y in range(int(max_year)-1, int(max_year)+1))
        if early_count == 0 and late_count == 0:
            growth = 0
        else:
            growth = (late_count - early_count) / (early_count + 1)
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
    """Bubble chart: x=degree, y=frequency, size=efficiency, color=category."""
    if len(valid_concepts) < 3:
        return
    freq = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
    degree = [nx_graph.degree(c) for c in valid_concepts]
    eff = [f * np.log1p(d) for f, d in zip(freq, degree)]
    categories = [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]
    df_bubble = pd.DataFrame({
        'concept': valid_concepts, 'frequency': freq, 'degree': degree,
        'efficiency': eff, 'category': categories
    })
    unique_cats = df_bubble['category'].unique()
    fig = px.scatter(df_bubble, x='degree', y='frequency', size='efficiency',
                     color='category', hover_name='concept',
                     title='Concept Landscape: Degree vs Frequency',
                     labels={'degree':'Degree (connectivity)', 'frequency':'Abstract Frequency'},
                     size_max=30,
                     color_discrete_sequence=get_colormap_colors('viridis', len(unique_cats)))
    st.plotly_chart(fig, use_container_width=True)


def render_degree_distribution(nx_graph):
    """Plot degree distribution with power-law fit."""
    degrees = [d for n, d in nx_graph.degree()]
    if not degrees:
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Degree Distribution", "Log-Log Degree Distribution"))

    # Histogram
    fig.add_trace(go.Histogram(x=degrees, nbinsx=30, marker_color='steelblue', name='Degree'), row=1, col=1)

    # Log-log
    degree_counts = Counter(degrees)
    x_vals = sorted(degree_counts.keys())
    y_vals = [degree_counts[x] for x in x_vals]
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', marker_color='crimson', name='Log-Log'), row=1, col=2)
    fig.update_xaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)

    fig.update_layout(title_text="Network Degree Distribution Analysis", showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_centrality_comparison(nx_graph, valid_concepts, concept_abstract_map):
    """Compare different centrality metrics side by side."""
    if nx_graph.number_of_nodes() < 3:
        return

    try:
        degree_cent = nx.degree_centrality(nx_graph)
        betweenness_cent = nx.betweenness_centrality(nx_graph, normalized=True, k=min(100, nx_graph.number_of_nodes()))
        closeness_cent = nx.closeness_centrality(nx_graph)
        eigenvector_cent = nx.eigenvector_centrality(nx_graph, max_iter=1000)

        df_cent = pd.DataFrame({
            'concept': list(nx_graph.nodes()),
            'degree': [degree_cent.get(n, 0) for n in nx_graph.nodes()],
            'betweenness': [betweenness_cent.get(n, 0) for n in nx_graph.nodes()],
            'closeness': [closeness_cent.get(n, 0) for n in nx_graph.nodes()],
            'eigenvector': [eigenvector_cent.get(n, 0) for n in nx_graph.nodes()],
            'frequency': [len(concept_abstract_map.get(n, [])) for n in nx_graph.nodes()]
        })

        # Correlation heatmap
        corr_cols = ['degree', 'betweenness', 'closeness', 'eigenvector', 'frequency']
        corr_matrix = df_cent[corr_cols].corr()

        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Centrality Correlation Matrix", "Top 15 by Eigenvector Centrality"),
                           specs=[[{"type": "heatmap"}, {"type": "bar"}]])

        fig.add_trace(go.Heatmap(z=corr_matrix.values, x=corr_cols, y=corr_cols,
                                 colorscale='RdBu', zmid=0), row=1, col=1)

        top_eigen = df_cent.nlargest(15, 'eigenvector')
        fig.add_trace(go.Bar(x=top_eigen['concept'], y=top_eigen['eigenvector'],
                            marker_color='teal'), row=1, col=2)
        fig.update_xaxes(tickangle=45, row=1, col=2)

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Centrality analysis failed: {e}")


# ==============================================================================
# SUNBURST & RADAR CHARTS
# ==============================================================================
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
    use_remainder = n_items > 80
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
    branchvalues = "remainder" if use_remainder else branchvalues
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


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================
def export_graph(nx_graph, concept_abstract_map, format_type: str):
    if format_type == "GraphML":
        try:
            nx.write_graphml_lxml(nx_graph, "nano_graph.graphml")
        except:
            nx.write_graphml(nx_graph, "nano_graph.graphml")
        with open("nano_graph.graphml", "rb") as f:
            return f.read(), "application/graphml+xml", "nano_graph.graphml"
    elif format_type == "GEXF":
        nx.write_gexf(nx_graph, "nano_graph.gexf")
        with open("nano_graph.gexf", "rb") as f:
            return f.read(), "application/xml", "nano_graph.gexf"
    elif format_type == "JSON":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "nano_graph.json"
    elif format_type == "CSV (Edges)":
        edge_data = []
        for u, v, data in nx_graph.edges(data=True):
            row = {"source": u, "target": v}
            row.update({k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))})
            edge_data.append(row)
        csv_df = pd.DataFrame(edge_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "nano_edges.csv"
    elif format_type == "CSV (Nodes)":
        node_data = []
        for node in nx_graph.nodes():
            row = {"concept": node, "frequency": len(concept_abstract_map.get(node, [])),
                   "degree": nx_graph.degree(node)}
            row.update({k: v for k, v in nx_graph.nodes[node].items()})
            node_data.append(row)
        csv_df = pd.DataFrame(node_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "nano_nodes.csv"
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
            return buf.read(), "image/png", "nano_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    elif format_type == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12))
            node_colors = [get_nanomaterials_category_color(n) for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                   node_size=400, font_size=7, font_weight='bold', edgecolors='white', linewidths=1)
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight', facecolor='white')
            buf.seek(0); plt.close()
            return buf.read(), "image/svg+xml", "nano_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None
    return None, None, None


# ==============================================================================
# GRAPH METRICS DASHBOARD
# ==============================================================================
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


# ==============================================================================
# THEME CONFIGURATION
# ==============================================================================
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
    "Off": {
        "damping": 0.99, "gravity": 0, "spring_length": 200,
        "spring_strength": 0.0, "central_gravity": 0.0, "stabilization": 0
    }
}

# ==============================================================================
# SIDEBAR CONFIGURATION (Enhanced with editing tools)
# ==============================================================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🎨 Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0
        )
        theme = THEME_PRESETS[st.session_state['theme']]

        st.subheader("🔬 Nanomaterials Focus Areas")
        st.markdown("- Core-shell Cu@Ag / Ag@Cu nanoparticles")
        st.markdown("- Bimetallic Ag-Cu nanostructures")
        st.markdown("- Synthesis (seed-mediated growth, galvanic replacement, co-reduction)")
        st.markdown("- Interfacial engineering (lattice mismatch, epitaxial growth, interdiffusion)")
        st.markdown("- Functional properties (plasmonics, SERS, catalysis, antibacterial)")
        st.markdown("- Characterization (HAADF-STEM, EDS mapping, UV-Vis, XRD peak shifts)")
        st.markdown("- Computational methods (FDTD, DDA, DFT, MD)")

        st.subheader("🖼️ Visualization")
        st.session_state['viz_backend'] = st.selectbox(
            "Engine:", ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"], index=0
        )
        st.session_state['cmap_name'] = st.selectbox(
            "Colormap:", options=list(SUPPORTED_COLORMAPS.keys()), index=0
        )

        st.subheader("🔧 Physics & Layout")
        st.session_state['physics_preset'] = st.selectbox(
            "Physics preset:",
            options=list(PHYSICS_PRESETS.keys()),
            index=0
        )
        preset = PHYSICS_PRESETS[st.session_state['physics_preset']]
        st.session_state['physics_enabled'] = st.checkbox(
            "Enable physics", value=(preset["gravity"] != 0)
        )

        with st.expander("⚙️ Advanced Physics Overrides"):
            st.session_state['adv_damping'] = st.slider("Damping", 0.05, 0.95, preset["damping"], step=0.05)
            st.session_state['adv_gravity'] = st.slider("Repulsion", -8000, -500, preset["gravity"], step=100)
            st.session_state['adv_spring_length'] = st.slider("Spring length", 40, 300, preset["spring_length"], step=10)
            st.session_state['adv_spring_strength'] = st.slider("Spring strength", 0.01, 0.20, preset["spring_strength"], step=0.01)
            st.session_state['adv_central_gravity'] = st.slider("Central gravity", 0.0, 0.5, preset["central_gravity"], step=0.05)
            st.session_state['adv_stabilization'] = st.slider("Stabilization iter", 0, 5000, preset["stabilization"], step=250)

        base_preset = PHYSICS_PRESETS[st.session_state['physics_preset']].copy()
        if st.session_state.get('adv_damping') is not None:
            base_preset["damping"] = st.session_state['adv_damping']
            base_preset["gravity"] = st.session_state['adv_gravity']
            base_preset["spring_length"] = st.session_state['adv_spring_length']
            base_preset["spring_strength"] = st.session_state['adv_spring_strength']
            base_preset["central_gravity"] = st.session_state['adv_central_gravity']
            base_preset["stabilization"] = st.session_state['adv_stabilization']
        st.session_state['effective_physics'] = base_preset

        st.subheader("📊 Display Limits")
        col_all1, col_slider1 = st.columns([0.3, 0.7])
        with col_all1:
            all_graph = st.checkbox("All", value=True, key="all_graph_chk")
        with col_slider1:
            st.session_state['top_n_graph'] = st.slider(
                "Max nodes", 10, 500, 200, step=10, disabled=all_graph,
                key="top_n_graph_slider"
            )
        if all_graph:
            st.session_state['top_n_graph'] = 0

        col_all2, col_slider2 = st.columns([0.3, 0.7])
        with col_all2:
            all_sun = st.checkbox("All", value=True, key="all_sun_chk")
        with col_slider2:
            st.session_state['top_n_sunburst'] = st.slider(
                "Max children/category", 10, 100, 40, step=10, disabled=all_sun,
                key="top_n_sunburst_slider"
            )
        if all_sun:
            st.session_state['top_n_sunburst'] = 0

        col_all3, col_slider3 = st.columns([0.3, 0.7])
        with col_all3:
            all_radar = st.checkbox("All", value=True, key="all_radar_chk")
        with col_slider3:
            st.session_state['top_n_radar'] = st.slider(
                "Top K for radar", 5, 30, 15, disabled=all_radar,
                key="top_n_radar_slider"
            )
        if all_radar:
            st.session_state['top_n_radar'] = 0

        st.subheader("🔧 Graph Parameters")
        st.session_state['min_freq'] = st.slider("Min concept frequency", 1, 20, 1)
        st.session_state['min_words'] = st.slider("Min words per concept", 2, 5, 2)
        st.session_state['sim_threshold'] = st.slider("Semantic threshold", 0.6, 0.95, 0.85, step=0.05)
        st.session_state['cooc_weight'] = st.slider("Co-occurrence weight", 0.5, 1.0, 0.9, step=0.1)
        st.session_state['sem_weight'] = st.slider("Semantic weight", 0.0, 0.5, 0.1, step=0.1)

        st.subheader("📐 Statistics")
        st.session_state['bootstrap_samples'] = st.slider("Bootstrap samples", 100, 2000, 500, step=100)
        st.session_state['alpha_level'] = st.selectbox("Significance α", [0.01, 0.05, 0.10], index=1)

        # ===== ENHANCED: Graph Editing Section =====
        st.subheader("✏️ Graph Editing")
        with st.expander("Edit Graph (Nodes/Edges)"):
            if 'analysis_data' in st.session_state and st.session_state.analysis_data is not None:
                data = st.session_state.analysis_data
                valid_concepts = data['valid_concepts']

                # Remove Nodes
                st.markdown("**Remove Nodes**")
                remove_selected = st.multiselect("Select nodes to remove", valid_concepts, key='remove_nodes')
                if st.button("Remove Selected Nodes", key='remove_btn'):
                    if remove_selected:
                        edits = {'remove_nodes': remove_selected}
                        G_new, map_new, concepts_new = apply_graph_edits(
                            data['nx_graph'], data['concept_abstract_map'], valid_concepts, edits
                        )
                        st.session_state.analysis_data['nx_graph'] = G_new
                        st.session_state.analysis_data['concept_abstract_map'] = map_new
                        st.session_state.analysis_data['valid_concepts'] = concepts_new
                        st.session_state.analysis_data['concept_to_id'] = {c: i for i, c in enumerate(concepts_new)}
                        st.session_state.analysis_data['id_to_concept'] = {i: c for i, c in enumerate(concepts_new)}
                        st.success(f"Removed {len(remove_selected)} nodes.")
                        st.rerun()
                    else:
                        st.warning("Select at least one node.")

                st.markdown("---")
                # Merge Nodes
                st.markdown("**Merge Nodes**")
                merge_list = st.multiselect("Select nodes to merge", valid_concepts, key='merge_nodes_select')
                new_name = st.text_input("New concept name", key='merge_new_name')
                if st.button("Merge Selected", key='merge_btn'):
                    if len(merge_list) >= 2 and new_name.strip():
                        edits = {'merge_nodes': [(merge_list, new_name.strip())]}
                        G_new, map_new, concepts_new = apply_graph_edits(
                            data['nx_graph'], data['concept_abstract_map'], valid_concepts, edits
                        )
                        st.session_state.analysis_data['nx_graph'] = G_new
                        st.session_state.analysis_data['concept_abstract_map'] = map_new
                        st.session_state.analysis_data['valid_concepts'] = concepts_new
                        st.session_state.analysis_data['concept_to_id'] = {c: i for i, c in enumerate(concepts_new)}
                        st.session_state.analysis_data['id_to_concept'] = {i: c for i, c in enumerate(concepts_new)}
                        st.success(f"Merged {len(merge_list)} nodes into '{new_name.strip()}'.")
                        st.rerun()
                    else:
                        st.warning("Select at least 2 nodes and provide a new name.")

                st.markdown("---")
                # Add Edge
                st.markdown("**Add Edge**")
                nodes = valid_concepts
                u = st.selectbox("From", nodes, key='edge_u')
                v = st.selectbox("To", nodes, key='edge_v')
                weight = st.number_input("Weight", value=1.0, min_value=0.1, key='edge_weight')
                if st.button("Add Edge", key='add_edge_btn'):
                    if u != v:
                        edits = {'add_edges': [(u, v, weight)]}
                        G_new, map_new, concepts_new = apply_graph_edits(
                            data['nx_graph'], data['concept_abstract_map'], valid_concepts, edits
                        )
                        st.session_state.analysis_data['nx_graph'] = G_new
                        st.session_state.analysis_data['concept_abstract_map'] = map_new
                        st.session_state.analysis_data['valid_concepts'] = concepts_new
                        st.success(f"Edge added: {u} -- {v} (weight={weight})")
                        st.rerun()
                    else:
                        st.warning("Source and target must be different.")

                st.markdown("---")
                # Filter by degree/frequency
                st.markdown("**Filter Graph**")
                min_deg = st.slider("Min degree", 0, 20, 0, key='filter_deg')
                min_freq_f = st.slider("Min frequency", 0, 50, 0, key='filter_freq')
                if st.button("Apply Filters", key='filter_btn'):
                    edits = {}
                    if min_deg > 0:
                        edits['filter_by_degree'] = min_deg
                    if min_freq_f > 0:
                        edits['filter_by_freq'] = min_freq_f
                    if edits:
                        G_new, map_new, concepts_new = apply_graph_edits(
                            data['nx_graph'], data['concept_abstract_map'], valid_concepts, edits
                        )
                        st.session_state.analysis_data['nx_graph'] = G_new
                        st.session_state.analysis_data['concept_abstract_map'] = map_new
                        st.session_state.analysis_data['valid_concepts'] = concepts_new
                        st.session_state.analysis_data['concept_to_id'] = {c: i for i, c in enumerate(concepts_new)}
                        st.session_state.analysis_data['id_to_concept'] = {i: c for i, c in enumerate(concepts_new)}
                        st.success("Filters applied.")
                        st.rerun()
            else:
                st.info("Build the graph first to enable editing.")

        # ===== ENHANCED: Sunburst filter =====
        st.subheader("☀️ Sunburst Options")
        st.session_state['sunburst_categories'] = st.multiselect(
            "Filter categories",
            options=list(CATEGORY_DISPLAY_NAMES.keys()),
            default=[],
            key='sunburst_cat_filter',
            format_func=lambda x: CATEGORY_DISPLAY_NAMES.get(x, x)
        )
        st.session_state['sunburst_branchvalues'] = st.selectbox(
            "Branch values mode", ['total', 'remainder'], index=0, key='sunburst_branch'
        )

        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"🖥️ Device: {gpu_info}")


# ==============================================================================
# MAIN FUNCTION (Fully Enhanced)
# ==============================================================================
def main():
    st.title("🔬 NanoGraph-Explorer: Core-Shell Ag-Cu Nanostructure Analytics")
    st.caption("Publication-ready concept graph analytics for Ag-Cu and Cu@Ag core-shell nanostructures • Plasmonics, Catalysis, Interfacial Engineering")
    render_sidebar()

    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = None
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = GraphEditHistory()

    # ─── LOAD JSON DATA ───
    st.header("📂 Data Loading")
    st.info(f"Place JSON/JSONL/CSV/TSV/BIB files in: `{JSON_METADATA_DIR}`")
    with st.spinner("Scanning json_metadatabase..."):
        file_records = load_all_json_files(JSON_METADATA_DIR)
        df = build_master_dataframe(file_records)
    if not file_records:
        st.warning("No supported files found in the directory.")
        st.info("Please place your metadata files in the `json_metadatabase/` folder.")
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

                # Compute advanced analytics
                st.write("🔬 Computing advanced analytics...")
                burst_df = detect_keyword_bursts(df_filtered, concept_abstract_map, valid_concepts)
                bridge_df = detect_cross_domain_bridges(nx_graph, valid_concepts)
                motif_data = analyze_network_motifs(nx_graph, valid_concepts)
                drift_df = detect_semantic_drift(valid_concepts, concept_abstract_map, all_texts, embed_model, df_filtered=df_filtered)
                genealogy_df = build_concept_genealogy(nx_graph, valid_concepts, concept_abstract_map)

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
                    "df_filtered": df_filtered,
                    "burst_df": burst_df,
                    "bridge_df": bridge_df,
                    "motif_data": motif_data,
                    "drift_df": drift_df,
                    "genealogy_df": genealogy_df
                }
                # Initialize edit history
                st.session_state.edit_history = GraphEditHistory()
                st.session_state.edit_history.push_state({
                    'nx_graph': nx_graph.copy(),
                    'concept_abstract_map': dict(concept_abstract_map),
                    'valid_concepts': list(valid_concepts)
                })
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

        # ===== ENHANCED: Seven tabs =====
        viz_tab, distill_tab, scores_tab, valid_tab, extra_tab, advanced_tab, export_tab = st.tabs([
            "🎨 Visualization", "📊 Distillation", "🎯 Research Directions", 
            "📐 Validation", "🌟 Extra Viz", "🔬 Advanced Analytics", "📥 Export"
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
                cat_filter = st.session_state.get('sunburst_categories', [])
                if cat_filter:
                    st.info(f"Filtering categories: {', '.join(cat_filter)}")
                branchval = st.session_state.get('sunburst_branchvalues', 'total')
                labels, parents, values = build_category_hierarchy(
                    valid_concepts, concept_abstract_map,
                    top_n_per_category=st.session_state.get('top_n_sunburst', 0),
                    categories_filter=cat_filter if cat_filter else None
                )
                render_sunburst_chart(labels, parents, values, cmap_name=cmap, theme=theme,
                                      branchvalues=branchval)

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

        # ===== ENHANCED: Extra Visualization Tab =====
        with extra_tab:
            st.subheader("🌟 Innovative Visualizations for Core-Shell Ag-Cu Concepts")

            df_filtered = data.get('df_filtered', pd.DataFrame())

            if not df_filtered.empty and 'Year' in df_filtered.columns:
                with st.expander("📅 Concept Timeline (Yearly Trends)", expanded=True):
                    render_timeline(df_filtered, concept_abstract_map, valid_concepts)

            with st.expander("🔥 Co-occurrence Heatmap"):
                top_n_heat = st.slider("Top N concepts for heatmap", 10, 50, 30, key='heat_top')
                render_cooccurrence_heatmap(nx_graph, valid_concepts, concept_abstract_map, top_n=top_n_heat)

            with st.expander("📊 t-SNE Projection"):
                render_tsne_projection(valid_concepts, data['embed_model'], concept_abstract_map, cmap)

            with st.expander("🔮 Community Detection (Modularity)"):
                render_community_detection(nx_graph, valid_concepts)

            if not df_filtered.empty and 'Year' in df_filtered.columns:
                with st.expander("📈 Concept Growth Rate"):
                    render_concept_growth(df_filtered, concept_abstract_map, valid_concepts)

            with st.expander("🫧 Concept Landscape (Degree vs Frequency)"):
                render_bubble_chart(valid_concepts, concept_abstract_map, nx_graph)

            with st.expander("📊 Degree Distribution"):
                render_degree_distribution(nx_graph)

            with st.expander("🏆 Centrality Comparison"):
                render_centrality_comparison(nx_graph, valid_concepts, concept_abstract_map)

        # ===== NEW: Advanced Analytics Tab =====
        with advanced_tab:
            st.subheader("🔬 Advanced Analytics & Discovery")

            with st.expander("💥 Keyword Burst Detection"):
                burst_df = data.get('burst_df', pd.DataFrame())
                if not burst_df.empty:
                    st.dataframe(burst_df.head(20), use_container_width=True)
                    fig = px.bar(burst_df.head(20), x='concept', y='burst_score',
                                color='max_consecutive',
                                title='Top Bursty Concepts',
                                labels={'burst_score': 'Burst Score'})
                    fig.update_layout(xaxis=dict(tickangle=45))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Burst detection requires Year data.")

            with st.expander("🌉 Cross-Domain Bridge Detection"):
                bridge_df = data.get('bridge_df', pd.DataFrame())
                if not bridge_df.empty:
                    st.dataframe(bridge_df.head(20), use_container_width=True)
                    fig = px.scatter(bridge_df, x='betweenness', y='category_diversity',
                                    size='neighbor_count', color='own_category',
                                    hover_name='concept',
                                    title='Bridge Concepts: Betweenness vs Category Diversity')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No bridge data available.")

            with st.expander("🧬 Network Motif Analysis"):
                motif_data = data.get('motif_data', {})
                if motif_data:
                    cols = st.columns(4)
                    cols[0].metric("Triangles", motif_data.get('triangles', 0))
                    cols[1].metric("Star Nodes", motif_data.get('stars', 0))
                    cols[2].metric("4-Cliques", motif_data.get('cliques_4', 0))
                    cols[3].metric("Transitivity", f"{motif_data.get('transitivity', 0):.4f}")

                    # Motif bar chart
                    motif_vals = {k: v for k, v in motif_data.items() if k in ['triangles', 'stars', 'cliques_4']}
                    if motif_vals:
                        fig = px.bar(x=list(motif_vals.keys()), y=list(motif_vals.values()),
                                    title='Network Motif Counts',
                                    labels={'x': 'Motif Type', 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No motif data available.")

            with st.expander("🔄 Semantic Drift Detection"):
                drift_df = data.get('drift_df', pd.DataFrame())
                if not drift_df.empty:
                    st.dataframe(drift_df.head(20), use_container_width=True)
                    fig = px.bar(drift_df.head(20), x='concept', y='semantic_drift',
                                color='late_count',
                                title='Concepts with Highest Semantic Drift',
                                labels={'semantic_drift': 'Drift Score'})
                    fig.update_layout(xaxis=dict(tickangle=45))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Semantic drift requires Year data and sufficient temporal coverage.")

            with st.expander("👨‍👩‍👧 Concept Genealogy"):
                genealogy_df = data.get('genealogy_df', pd.DataFrame())
                if not genealogy_df.empty:
                    st.dataframe(genealogy_df[['concept', 'generation', 'n_parents', 'n_children', 'n_siblings', 'frequency']].head(20),
                                use_container_width=True)
                    fig = px.scatter(genealogy_df, x='generation', y='frequency',
                                    size='n_children', color='n_parents',
                                    hover_name='concept',
                                    title='Concept Genealogy: Generation vs Frequency')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No genealogy data available.")

            with st.expander("📄 Generate Analysis Report"):
                if st.button("Generate Full Report"):
                    report = generate_analysis_report(
                        data, compute_graph_metrics(nx_graph),
                        validate_graph_metrics(nx_graph, valid_concepts),
                        top_scores, distill_df,
                        data.get('burst_df', pd.DataFrame()),
                        data.get('bridge_df', pd.DataFrame()),
                        data.get('motif_data', {}),
                        data.get('drift_df', pd.DataFrame()),
                        data.get('genealogy_df', pd.DataFrame())
                    )
                    st.markdown(report)
                    st.download_button("📥 Download Report (MD)", data=report.encode('utf-8'),
                                      file_name="nanograph_analysis_report.md", mime="text/markdown")

        with export_tab:
            st.subheader("📥 Export & Post-Processing")

            col1, col2 = st.columns(2)
            with col1:
                export_format = st.selectbox("Network Format:", 
                    ["GraphML", "GEXF", "JSON", "CSV (Edges)", "CSV (Nodes)"])
                if st.button("📤 Export Network"):
                    result = export_graph(nx_graph, concept_abstract_map, export_format)
                    if result[0]:
                        data_bytes, mime, filename = result
                        st.download_button("💾 Save Network", data=data_bytes, file_name=filename, mime=mime)

            with col2:
                fig_format = st.selectbox("Figure Format:", ["PNG", "SVG"])
                fig_type = st.selectbox("Figure Type:", ["network", "degree_distribution", "community"])
                dpi = st.slider("DPI", 150, 600, 300, step=50)
                if st.button("📤 Export Figure"):
                    fig_bytes = export_publication_figure(
                        nx_graph, concept_abstract_map, valid_concepts,
                        figure_type=fig_type, dpi=dpi, cmap_name=cmap
                    )
                    mime = "image/png" if fig_format == "PNG" else "image/svg+xml"
                    ext = "png" if fig_format == "PNG" else "svg"
                    st.download_button(f"💾 Save {fig_format}", data=fig_bytes,
                                      file_name=f"nano_graph_{fig_type}.{ext}", mime=mime)

            concept_list_df = pd.DataFrame({
                'concept': valid_concepts,
                'frequency': [len(concept_abstract_map.get(c, [])) for c in valid_concepts],
                'degree': [nx_graph.degree(c) for c in valid_concepts],
                'category': [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]
            })
            csv_concepts = concept_list_df.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download Concept List (CSV)", data=csv_concepts,
                              file_name="concepts.csv", mime="text/csv")

            # Export all advanced analytics
            if not data.get('burst_df', pd.DataFrame()).empty:
                csv_burst = data['burst_df'].to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download Burst Data (CSV)", data=csv_burst,
                                  file_name="burst_analysis.csv", mime="text/csv")
            if not data.get('bridge_df', pd.DataFrame()).empty:
                csv_bridge = data['bridge_df'].to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download Bridge Data (CSV)", data=csv_bridge,
                                  file_name="bridge_analysis.csv", mime="text/csv")

if __name__ == "__main__":
    main()
