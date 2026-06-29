#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nanomaterials-ConceptGraph: Concept Graph Builder for Defect-Engineered Ag Nanoparticles & Applications
==================================================================================
Large-corpus concept graph extraction (3000+ abstracts) from JSON metadata.
No seed injection needed — robust statistical methods for high-volume data.

Features:
- Robust JSON/JSONL/CSV loading with BOM handling and error recovery
- Large-corpus optimized concept extraction (TF-IDF, semantic clustering, PageRank)
- Defect-Ag focused domain filtering (ISF, EST, twins, SERS, catalysis, sensing, etc.)
- Interactive PyVis/Plotly 2D/3D visualizations with 50+ colormaps
- Statistical validation: modularity, silhouette, centrality, bootstrap CIs
- GNN-powered (GraphSAGE) research direction scoring with PyTorch
- Export: GraphML, JSON, CSV, HTML, SVG, PNG
- Streamlit UI with session persistence and crash prevention

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn

Run: streamlit run nanomaterials_concept_graph.py

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
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import davies_bouldin_score, pairwise_distances
from sklearn.decomposition import PCA
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
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Nanomaterials-ConceptGraph: Defect-Engineered Ag Explorer",
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
# COLORMAP REGISTRY (50+)
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
    """Convert matplotlib colormap to list of hex colors for Plotly/PyVis"""
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
# ROBUST JSON LOADER
# ==========================================
def robust_load_file(filepath: Path):
    """Try multiple strategies to load a file that claims to be JSON."""
    text = filepath.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"File is empty (0 bytes or only whitespace).")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    sanitized = re.sub(r'\bNaN\b', 'null', text)
    sanitized = re.sub(r'\bInfinity\b', 'null', sanitized)
    sanitized = re.sub(r'-Infinity', 'null', sanitized)
    sanitized = re.sub(r',(\s*[}\]])', r'\1', sanitized)
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
    """Load every .json file in directory and return a list of (filepath, records)."""
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
                st.code(f"Hex preview (first {len(raw_bytes)} bytes):\n{formatted}", language="text")
            except Exception:
                pass
    return loaded

@st.cache_data(show_spinner=False)
def build_master_dataframe(file_records):
    """Flatten all records into one DataFrame."""
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
# DEFECT-ENGINEERED AG DOMAIN CONFIGURATION
# ==========================================
# Core materials of interest
CORE_MATERIALS = [
    "defect engineered silver", "defect engineered ag", "defective ag nanoparticle", "defective silver nanoparticle",
    "twinned ag nanoparticle", "twinned silver nanoparticle", "nanotwinned ag", "nanotwinned silver",
    "stacking faulted ag", "stacking faulted silver", "ag nanoparticle with isf", "ag nanoparticle with est",
    "intrinsic stacking fault ag", "extrinsic stacking fault ag", "isf ag", "est ag",
    "point defect ag", "vacancy rich ag", "dislocation rich ag", "ag nanocrystal defect",
    "silver nanowire defect", "ag nanostructure defect", "defect rich silver", "defect rich ag"
]

# Physicochemical properties (Replacing bulk mechanical properties)
PHYSICOCHEMICAL_PROPERTIES = [
    "defect formation energy", "stacking fault energy", "sfe", "surface energy", "cohesive energy",
    "binding energy", "adsorption energy", "activation energy", "d-band center",
    "melting point", "thermal stability", "structural stability", "coalescence",
    "surface plasmon resonance", "spr", "localized surface plasmon resonance", "lspr",
    "plasmon peak", "extinction coefficient", "absorption cross section", "hot spot",
    "enhancement factor", "sers efficiency", "catalytic activity", "turnover frequency", "tof",
    "overpotential", "current density", "onset potential", "exchange current density",
    "band gap", "work function", "fermi level", "schottky barrier",
    "antibacterial activity", "minimum inhibitory concentration", "mic", "cytotoxicity"
]

# Synthesis and defect introduction
SYNTHESIS_METHODS = [
    "chemical reduction", "polyol method", "seed mediated growth", "seed-mediated",
    "galvanic replacement", "galvanic displacement", "co-reduction",
    "electrochemical synthesis", "electrodeposition", "template assisted",
    "plasma treatment", "plasma irradiation", "ion irradiation", "electron irradiation", "gamma irradiation",
    "laser ablation", "photoreduction", "sonochemical", "microwave assisted",
    "thermal annealing", "defect engineering", "defect introduction", "strain engineering",
    "capping agent", "surfactant", "pvp", "ctab", "oleylamine", "oleic acid", "citrate",
    "ligand exchange", "surface functionalization", "green synthesis"
]

# Structure and characterization
STRUCTURE_CHARACTERIZATION = [
    "high resolution tem", "hrtem", "scanning transmission electron microscopy", "stem", "haadf-stem",
    "energy dispersive x-ray spectroscopy", "eds", "electron energy loss spectroscopy", "eels",
    "selected area electron diffraction", "saed", "x-ray diffraction", "xrd",
    "stacking fault probability", "twin probability", "xrd peak shift", "xrd peak broadening",
    "uv-vis spectroscopy", "uv-vis-nir", "extinction spectrum", "absorption spectrum",
    "surface enhanced raman scattering", "sers", "sers mapping", "raman spectroscopy",
    "atomic force microscopy", "afm", "scanning electron microscopy", "sem",
    "x-ray photoelectron spectroscopy", "xps", "dynamic light scattering", "dls",
    "defect density", "twin spacing", "twin boundary density", "fringe analysis", "moiré pattern",
    "strain field", "lattice distortion", "interplanar spacing", "d-spacing"
]

# Computational methods
COMPUTATIONAL_METHODS = [
    "density functional theory", "dft", "ab initio", "first principles",
    "finite difference time domain", "fdtd", "discrete dipole approximation", "dda", "mie theory",
    "boundary element method", "bem", "finite element method", "fem",
    "molecular dynamics", "md", "ab initio md", "aimd",
    "comsol multiphysics", "lumerical", "meep",
    "adsorption energy calculation", "transition state", "neb", "nudged elastic band",
    "machine learning", "neural network", "high throughput screening"
]

# Functional properties and technological applications
FUNCTIONAL_PROPERTIES = [
    "surface enhanced raman scattering", "sers", "sers substrate", "sers activity", "hot spot",
    "electromagnetic enhancement", "chemical enhancement", "single molecule detection",
    "catalytic activity", "electrocatalysis", "photocatalysis", "plasmonic catalysis",
    "co2 reduction", "co2rr", "oxygen reduction reaction", "orr", "oxygen evolution reaction", "oer",
    "hydrogen evolution reaction", "her", "nitrate reduction", "nitrite reduction",
    "antibacterial", "antimicrobial", "bactericidal", "antibiofilm", "wound healing",
    "biosensor", "chemical sensor", "gas sensor", "colorimetric sensor", "sensing",
    "photothermal therapy", "ptt", "photodynamic therapy", "pdt", "theranostics", "bioimaging",
    "localized surface plasmon resonance", "lspr", "refractive index sensitivity"
]

ALL_DOMAIN_KEYWORDS = (CORE_MATERIALS + PHYSICOCHEMICAL_PROPERTIES + SYNTHESIS_METHODS +
                       STRUCTURE_CHARACTERIZATION + COMPUTATIONAL_METHODS + FUNCTIONAL_PROPERTIES)

# Regex patterns for nanomaterials concept extraction
NANOMATERIALS_PATTERNS = [
    r'\b(?:defect\s*(?:engineered|rich|abundant)\s*(?:ag|silver|nanoparticle|nanocrystal))\b',
    r'\b(?:nanotwinned\s*(?:ag|silver))\b',
    r'\b(?:twinned\s*(?:ag|silver|nanoparticle|nanocrystal))\b',
    r'\b(?:intrinsic\s*stacking\s*fault|isf|extrinsic\s*stacking\s*fault|est)\b',
    r'\b(?:stacking\s*faulted\s*(?:ag|silver|nanoparticle))\b',
    r'\b(?:ag\s*nanoparticle|silver\s*nanoparticle|ag\s*nanocrystal|silver\s*nanocrystal)\b',
    r'\b(?:sers|surface\s*enhanced\s*raman|hot\s*spot|electromagnetic\s*enhancement)\b',
    r'\b(?:catalytic|electrocatalysis|photocatalysis|co2rr|orr|oer|her)\b',
    r'\b(?:antibacterial|antimicrobial|bactericidal|theranostics|bioimaging)\b',
    r'\b(?:biosensor|chemical\s*sensor|gas\s*sensor|colorimetric\s*sensor)\b',
    r'\b(?:plasma\s*treatment|ion\s*irradiation|electron\s*irradiation|gamma\s*irradiation)\b',
    r'\b(?:defect\s*engineering|defect\s*introduction|strain\s*engineering)\b',
    r'\b(?:hrtem|stem|haadf|eels|xrd|saed|uv-vis|xps|dls)\b',
    r'\b(?:dft|fdtd|dda|md|molecular\s*dynamics|comsol|lumerical)\b',
    r'\b(?:\d+(?:\.\d+)?\s*(?:nm|µm|micrometer|angstrom|å|mv|v|ma/cm2|a/dm2))\b',
    r'\b(?:enhancement\s*factor|ef|overpotential|current\s*density|plasmon\s*peak)\b'
]

# Category mapping for concept classification
NANOMATERIALS_CATEGORY_MAPPING = {
    r'defect\s*(?:engineered|rich|abundant)\s*(?:ag|silver)|defective\s*(?:ag|silver)|vacancy\s*(?:rich\s*)?(?:ag|silver)': 'defect_engineered_ag',
    r'nanotwinned\s*(?:ag|silver)|twinned\s*(?:ag|silver)|twin\s*(?:boundary|boundaries|density|spacing|probability)': 'twinned_ag',
    r'intrinsic\s*stacking\s*fault|isf|extrinsic\s*stacking\s*fault|est|stacking\s*faulted\s*(?:ag|silver)': 'stacking_faults',
    r'point\s*defect|dislocation|line\s*defect|planar\s*defect|surface\s*defect': 'crystallographic_defects',
    r'sers|surface\s*enhanced\s*raman|hot\s*spot|electromagnetic\s*enhancement': 'sers_plasmonics',
    r'catalytic|electrocatalysis|photocatalysis|co2rr|orr|oer|her|reduction|oxidation': 'catalysis_application',
    r'antibacterial|antimicrobial|bactericidal|cytotoxicity|wound\s*healing|bioimaging|theranostics': 'biomedical_application',
    r'biosensor|chemical\s*sensor|gas\s*sensor|colorimetric|sensing': 'sensing_application',
    r'chemical\s*reduction|polyol|seed\s*mediated|galvanic|plasma|irradiation|defect\s*engineering': 'synthesis_defect_intro',
    r'hrtem|stem|haadf|eels|xrd|saed|uv-vis|xps|dls': 'advanced_characterization',
    r'dft|fdtd|dda|md|molecular\s*dynamics|comsol|lumerical': 'computational_modeling'
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
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

# ==========================================
# DEVICE & MODEL MANAGEMENT
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# ==========================================
# CONCEPT EXTRACTION & NORMALIZATION
# ==========================================
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
    # Normalize core materials & defects
    concept = re.sub(r'\bdefect\s*engineered\s*silver\b', 'defect engineered ag', concept)
    concept = re.sub(r'\bdefective\s*ag\s*nanoparticle\b', 'defect engineered ag', concept)
    concept = re.sub(r'\bnanotwinned\s*ag\b', 'nanotwinned ag', concept)
    concept = re.sub(r'\bintrinsic\s*stacking\s*fault\b', 'isf', concept)
    concept = re.sub(r'\bextrinsic\s*stacking\s*fault\b', 'est', concept)
    concept = re.sub(r'\bag\s*nanoparticle\b', 'ag nanoparticle', concept)
    concept = re.sub(r'\bsilver\s*nanoparticle\b', 'ag nanoparticle', concept)
    
    # Normalize characterization
    concept = re.sub(r'\bhigh\s*resolution\s*tem\b', 'hrtem', concept)
    concept = re.sub(r'\bscanning\s*transmission\s*electron\s*microscopy\b', 'stem', concept)
    concept = re.sub(r'\benergy\s*dispersive\s*x-ray\b', 'eds', concept)
    concept = re.sub(r'\belectron\s*energy\s*loss\s*spectroscopy\b', 'eels', concept)
    concept = re.sub(r'\bx-ray\s*diffraction\b', 'xrd', concept)
    concept = re.sub(r'\bx-ray\s*photoelectron\s*spectroscopy\b', 'xps', concept)
    concept = re.sub(r'\buv-vis\s*spectroscopy\b', 'uv-vis', concept)
    concept = re.sub(r'\bsurface\s*enhanced\s*raman\s*scattering\b', 'sers', concept)
    
    # Normalize computational
    concept = re.sub(r'\bdensity\s*functional\s*theory\b', 'dft', concept)
    concept = re.sub(r'\bab\s*initio\b', 'ab initio', concept)
    concept = re.sub(r'\bfirst\s*principles\b', 'first principles', concept)
    concept = re.sub(r'\bmolecular\s*dynamics\b', 'molecular dynamics', concept)
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
    material_prop_pattern = r'\b([A-Z][a-z]+(?:\d+(?:\.\d+)?)?(?:[\s\-][A-Z][a-z]?\d*)+)\b\s+(?:with|having|exhibiting|showing|demonstrating|achieving|reaching|delivering|providing|offering)\s+(?:a\s+)?([\d\.]+\s*(?:nm|um|µm|angstrom|å|mv|v|ma/cm2|a/dm2|ma\s*mg-1))\b'
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
        # Extract defect and application specific metrics
        size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:nm|um|µm)', combined_text, re.I)
        if size_matches: metrics['size_nm_um'] = [float(m) for m in size_matches]
        
        defect_density_matches = re.findall(r'(?:defect\s*density|twin\s*density)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:x\s*10\^\d+|e\+?\d+)?\s*(?:m\^-2|/m\^2|per\s*m\^2|cm\^-2)', combined_text, re.I)
        if defect_density_matches: metrics['defect_density'] = [float(m) for m in defect_density_matches]
        
        enhancement_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:x|×|\*)?\s*(?:enhancement\s*factor|ef|sers\s*ef)', combined_text, re.I)
        if enhancement_matches: metrics['sers_enhancement_factor'] = [float(m) for m in enhancement_matches]
        
        wavelength_matches = re.findall(r'(?:plasmon\s*peak|lspr\s*peak|absorption\s*peak)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:nm)', combined_text, re.I)
        if wavelength_matches: metrics['plasmon_peak_nm'] = [float(m) for m in wavelength_matches]
        
        overpotential_matches = re.findall(r'(?:overpotential|onset\s*potential)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:mv|v)', combined_text, re.I)
        if overpotential_matches: metrics['overpotential_mv'] = [float(m) for m in overpotential_matches]
        
        current_density_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ma/cm2|ma\s*cm-2|a/dm2|a\s*dm-2|ma\s*mg-1)', combined_text, re.I)
        if current_density_matches: metrics['current_density'] = [float(m) for m in current_density_matches]
        
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
            if any(re.search(p, concept, re.I) for p in [r'\bdefect\s*engineered', r'\bdefective', r'\bag\s*defect']):
                concept_to_abstract[concept] = 'defect_engineered_ag'
            elif any(re.search(p, concept, re.I) for p in [r'\bnanotwinned', r'\btwin', r'\bisf', r'\best', r'\bstacking\s*fault']):
                concept_to_abstract[concept] = 'twinned_ag'
            elif any(re.search(p, concept, re.I) for p in [r'\bsers', r'\bplasmon', r'\bhot\s*spot']):
                concept_to_abstract[concept] = 'sers_plasmonics'
            elif any(re.search(p, concept, re.I) for p in [r'\bcatalytic', r'\bco2rr', r'\borr', r'\boer', r'\bher']):
                concept_to_abstract[concept] = 'catalysis_application'
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
# GRAPH CONSTRUCTION
# ==========================================
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

# ==========================================
# GNN MODEL
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
# VISUALIZATION FUNCTIONS
# ==========================================
def get_nanomaterials_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    # Core materials & Defects
    if any(c in concept_lower for c in ['defect engineered ag', 'defective ag', 'vacancy rich ag']):
        return "#D32F2F"  # Red
    elif any(c in concept_lower for c in ['nanotwinned ag', 'twinned ag', 'twin boundary', 'twin density']):
        return "#E91E63"  # Pink
    elif any(c in concept_lower for c in ['isf', 'est', 'stacking fault', 'stacking faulted']):
        return "#9C27B0"  # Purple
    elif any(c in concept_lower for c in ['point defect', 'dislocation', 'crystallographic defect']):
        return "#FF9800"  # Orange
    # Applications
    elif any(c in concept_lower for c in ['sers', 'hot spot', 'plasmon', 'lspr']):
        return "#F44336"  # Red-Orange
    elif any(c in concept_lower for c in ['catalytic', 'electrocatalysis', 'photocatalysis', 'co2rr', 'orr', 'oer', 'her']):
        return "#00E676"  # Bright Green
    elif any(c in concept_lower for c in ['antibacterial', 'antimicrobial', 'bioimaging', 'theranostics', 'wound']):
        return "#8E24AA"  # Deep Purple
    elif any(c in concept_lower for c in ['biosensor', 'chemical sensor', 'gas sensor', 'sensing']):
        return "#18FFFF"  # Cyan accent
    # Synthesis & Characterization
    elif any(c in concept_lower for c in ['chemical reduction', 'polyol', 'seed mediated', 'galvanic', 'plasma', 'irradiation']):
        return "#00BCD4"  # Cyan
    elif any(c in concept_lower for c in ['hrtem', 'stem', 'haadf', 'eels', 'xrd', 'saed', 'uv-vis', 'xps']):
        return "#3F51B5"  # Indigo
    # Computational
    elif any(c in concept_lower for c in ['dft', 'fdtd', 'dda', 'md', 'molecular dynamics', 'comsol', 'lumerical']):
        return "#4CAF50"  # Green
    else:
        return "#9E9E9E"  # Grey

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
                          file_name="defect_ag_concept_graph.html", mime="text/html")
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

# ==========================================
# SUNBURST & RADAR CHARTS
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

def render_sunburst_chart(labels, parents, values, cmap_name="viridis", label_size=11, width=800, height=600, theme=None):
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
    branchvalues = "remainder" if use_remainder else "total"
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
        title="<b>Defect-Engineered Ag Nanoparticles & Applications Domain Hierarchy</b><br><i>Size = concept frequency</i>",
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

# ==========================================
# EXPORT FUNCTIONS
# ==========================================
def export_graph(nx_graph, concept_abstract_map, format_type: str):
    if format_type == "GraphML":
        try:
            nx.write_graphml_lxml(nx_graph, "nano_graph.graphml")
        except:
            nx.write_graphml(nx_graph, "nano_graph.graphml")
        with open("nano_graph.graphml", "rb") as f:
            return f.read(), "application/graphml+xml", "nano_graph.graphml"
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

# ==========================================
# INTERACTIVE GRAPH EDITING FUNCTIONS
# ==========================================
def apply_graph_edits(nx_graph, valid_concepts, concept_to_id, id_to_concept, 
                       concept_abstract_map, all_concepts, edit_operations):
    """
    Apply graph editing operations and return updated structures.

    edit_operations: list of dicts with keys 'op', 'nodes', 'new_name', 'weight'
    """
    modified = False
    for op in edit_operations:
        if op['op'] == 'remove':
            nodes_to_remove = [n for n in op['nodes'] if n in nx_graph]
            for n in nodes_to_remove:
                nx_graph.remove_node(n)
                if n in valid_concepts:
                    valid_concepts.remove(n)
                if n in concept_to_id:
                    del concept_to_id[n]
                if n in concept_abstract_map:
                    del concept_abstract_map[n]
            # Rebuild id_to_concept
            id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
            modified = True

        elif op['op'] == 'merge':
            nodes_to_merge = [n for n in op['nodes'] if n in nx_graph]
            new_name = op.get('new_name', ' + '.join(nodes_to_merge))
            if not nodes_to_merge or new_name in nx_graph:
                continue
            # Collect all edges and abstract mappings
            all_edges = {}
            all_abstracts = []
            for n in nodes_to_merge:
                for neighbor in nx_graph.neighbors(n):
                    if neighbor not in nodes_to_merge:
                        w = nx_graph[n][neighbor].get('weight', 1)
                        et = nx_graph[n][neighbor].get('edge_type', 'unknown')
                        if neighbor in all_edges:
                            all_edges[neighbor]['weight'] += w
                            all_edges[neighbor]['cooccurrence'] += nx_graph[n][neighbor].get('cooccurrence', 0)
                        else:
                            all_edges[neighbor] = {'weight': w, 'cooccurrence': nx_graph[n][neighbor].get('cooccurrence', 0),
                                                     'semantic': nx_graph[n][neighbor].get('semantic', 0), 'edge_type': et}
                all_abstracts.extend(concept_abstract_map.get(n, []))
            # Remove old nodes
            for n in nodes_to_merge:
                nx_graph.remove_node(n)
                if n in valid_concepts:
                    valid_concepts.remove(n)
                if n in concept_to_id:
                    del concept_to_id[n]
                if n in concept_abstract_map:
                    del concept_abstract_map[n]
            # Add new merged node
            nx_graph.add_node(new_name, frequency=len(set(all_abstracts)))
            valid_concepts.append(new_name)
            concept_abstract_map[new_name] = list(set(all_abstracts))
            # Add edges
            for neighbor, edge_data in all_edges.items():
                if neighbor in nx_graph:
                    nx_graph.add_edge(new_name, neighbor, **edge_data)
            # Rebuild mappings
            concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
            id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
            modified = True

        elif op['op'] == 'add_edge':
            u, v = op['nodes']
            weight = op.get('weight', 1.0)
            if u in nx_graph and v in nx_graph and not nx_graph.has_edge(u, v):
                nx_graph.add_edge(u, v, weight=weight, cooccurrence=0, semantic=0, edge_type='manual')
                modified = True

        elif op['op'] == 'rename':
            old_name, new_name = op['nodes'][0], op.get('new_name', '')
            if old_name in nx_graph and new_name and new_name not in nx_graph:
                nx_graph = nx.relabel_nodes(nx_graph, {old_name: new_name})
                idx = valid_concepts.index(old_name)
                valid_concepts[idx] = new_name
                concept_abstract_map[new_name] = concept_abstract_map.pop(old_name, [])
                concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
                id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
                modified = True

    return nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map, modified


def filter_graph_by_metrics(nx_graph, concept_abstract_map, min_degree=0, min_freq=0, 
                             max_degree=None, max_freq=None):
    """Filter graph nodes by degree and frequency thresholds."""
    nodes_to_keep = []
    for node in nx_graph.nodes():
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        if deg >= min_degree and freq >= min_freq:
            if max_degree is None or deg <= max_degree:
                if max_freq is None or freq <= max_freq:
                    nodes_to_keep.append(node)
    if len(nodes_to_keep) < 2:
        return nx_graph
    return nx_graph.subgraph(nodes_to_keep).copy()


# ==========================================
# ENHANCED SUNBURST FUNCTIONS
# ==========================================
def build_category_hierarchy_filtered(valid_concepts, concept_abstract_map, 
                                       selected_categories=None, top_n_per_category=40):
    """Build hierarchy with optional category filtering."""
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    category_map = abstract_concepts_to_categories(valid_concepts)
    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
        if selected_categories and category not in selected_categories:
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


def render_sunburst_chart_enhanced(labels, parents, values, cmap_name="viridis", 
                                    label_size=11, width=800, height=600, 
                                    branchvalues="total", theme=None):
    """Enhanced sunburst with branch values toggle."""
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return
    n_items = len(labels)
    use_remainder = branchvalues == "remainder" or n_items > 80
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
    bv = "remainder" if use_remainder else branchvalues
    fig = go.Figure(go.Sunburst(
        labels=unique_ids, parents=parent_ids, values=values, ids=unique_ids,
        branchvalues=bv,
        marker=dict(colors=colors, line=dict(width=0.5, color="white")),
        textinfo="label+percent entry+value",
        insidetextorientation="radial",
        textfont=dict(size=label_size),
        hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Parent: %{parent}<extra></extra>'
    ))
    fig.update_layout(
        title="<b>Defect-Engineered Ag Nanoparticles & Applications Domain Hierarchy</b><br><i>Size = concept frequency</i>",
        font=dict(size=label_size, family="Arial"),
        paper_bgcolor="white", plot_bgcolor="white",
        width=width, height=height,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# INNOVATIVE VISUALIZATION FUNCTIONS
# ==========================================
def render_concept_timeline(df, valid_concepts, concept_abstract_map, top_n=15, theme=None):
    """Render concept frequency timeline if Year data exists."""
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if 'Year' not in df.columns or df['Year'].isna().all():
        st.info("No valid Year column found in data. Timeline requires publication year information.")
        return

    df_year = df.dropna(subset=['Year']).copy()
    df_year['Year'] = pd.to_numeric(df_year['Year'], errors='coerce')
    df_year = df_year.dropna(subset=['Year'])
    if len(df_year) == 0:
        st.info("No valid year data available for timeline.")
        return

    # Get top concepts by frequency
    top_concepts = sorted(valid_concepts, 
                          key=lambda c: len(concept_abstract_map.get(c, [])), 
                          reverse=True)[:top_n]

    text_cols = [c for c in df_year.columns if any(k in c.lower() for k in ['abstract', 'title', 'summary', 'text'])]
    if not text_cols:
        st.info("No text columns found for timeline analysis.")
        return

    timeline_data = []
    for _, row in df_year.iterrows():
        year = int(row['Year'])
        text = " ".join([str(row.get(col, '')) for col in text_cols if pd.notna(row.get(col))])
        text_lower = text.lower()
        for concept in top_concepts:
            count = len(re.findall(r'\b' + re.escape(concept.lower()) + r'\b', text_lower))
            if count > 0:
                timeline_data.append({'Year': year, 'Concept': concept, 'Count': count})

    if not timeline_data:
        st.info("No concept occurrences found with year data.")
        return

    timeline_df = pd.DataFrame(timeline_data)
    timeline_agg = timeline_df.groupby(['Year', 'Concept'])['Count'].sum().reset_index()

    fig = px.line(timeline_agg, x='Year', y='Count', color='Concept',
                  title='Concept Frequency Over Time',
                  labels={'Count': 'Mentions', 'Year': 'Publication Year'},
                  template='plotly_white' if theme == THEME_PRESETS["Bright (Default)"] else 'plotly_dark')
    fig.update_layout(
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
        font_color=theme.get('font', '#000000'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cooccurrence_heatmap(nx_graph, valid_concepts, concept_abstract_map, 
                                 top_n=30, cmap_name="viridis", theme=None):
    """Render co-occurrence heatmap for top concepts."""
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]

    top_concepts = sorted(valid_concepts, 
                          key=lambda c: len(concept_abstract_map.get(c, [])), 
                          reverse=True)[:top_n]

    if len(top_concepts) < 3:
        st.info("Not enough concepts for heatmap.")
        return

    # Build co-occurrence matrix
    cooc_matrix = np.zeros((len(top_concepts), len(top_concepts)))
    concept_idx = {c: i for i, c in enumerate(top_concepts)}

    for u, v, data in nx_graph.edges(data=True):
        if u in concept_idx and v in concept_idx:
            i, j = concept_idx[u], concept_idx[v]
            w = data.get('cooccurrence', data.get('weight', 1))
            cooc_matrix[i][j] = w
            cooc_matrix[j][i] = w

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cooc_matrix,
        x=[c[:20] + "…" if len(c) > 20 else c for c in top_concepts],
        y=[c[:20] + "…" if len(c) > 20 else c for c in top_concepts],
        colorscale=cmap_name if cmap_name in px.colors.named_colorscales() else 'Viridis',
        hoverongaps=False,
        hovertemplate='%{x} ↔ %{y}<br>Co-occurrence: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title="Concept Co-occurrence Heatmap",
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        height=max(400, len(top_concepts) * 15),
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
        font_color=theme.get('font', '#000000')
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tsne_projection(valid_concepts, concept_abstract_map, embed_model, 
                           cmap_name="viridis", theme=None):
    """Render t-SNE projection of concept embeddings."""
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 5:
        st.info("Need at least 5 concepts for t-SNE projection.")
        return

    with st.spinner("Computing t-SNE projection..."):
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        perplexity = min(30, len(valid_concepts) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                    n_iter=1000, learning_rate='auto', init='pca')
        coords = tsne.fit_transform(embeddings)

    category_map = abstract_concepts_to_categories(valid_concepts)
    categories = [category_map.get(c, 'general') for c in valid_concepts]
    frequencies = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]

    df_tsne = pd.DataFrame({
        'x': coords[:, 0], 'y': coords[:, 1],
        'concept': valid_concepts, 'category': categories,
        'frequency': frequencies
    })

    fig = px.scatter(df_tsne, x='x', y='y', color='category', size='frequency',
                     hover_data=['concept', 'frequency'],
                     title='t-SNE Projection of Concept Embeddings',
                     color_discrete_sequence=get_colormap_colors(cmap_name, len(set(categories))),
                     template='plotly_white' if theme == THEME_PRESETS["Bright (Default)"] else 'plotly_dark')
    fig.update_layout(
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
        font_color=theme.get('font', '#000000'),
        height=600
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)


def render_community_detection(nx_graph, valid_concepts, concept_abstract_map, 
                                cmap_name="viridis", theme=None):
    """Detect and visualize communities with interactive coloring."""
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if nx_graph.number_of_nodes() < 5:
        st.info("Need at least 5 nodes for community detection.")
        return

    try:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(nx_graph))
    except Exception as e:
        st.warning(f"Community detection failed: {e}")
        return

    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    # Assign colors
    colors = get_colormap_colors(cmap_name, len(communities))

    pos = nx.spring_layout(nx_graph, seed=42, k=1.5)

    edge_x, edge_y = [], []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=0.8, color=theme.get('edge_unknown', '#cccccc')),
                            hoverinfo='skip')

    # Group nodes by community for legend
    community_traces = []
    for i, comm in enumerate(communities):
        comm_nodes = [n for n in comm if n in pos]
        if not comm_nodes:
            continue
        x_vals = [pos[n][0] for n in comm_nodes]
        y_vals = [pos[n][1] for n in comm_nodes]
        freqs = [len(concept_abstract_map.get(n, [])) for n in comm_nodes]
        texts = [f"{n}<br>Community: {i+1}<br>Freq: {len(concept_abstract_map.get(n, []))}" for n in comm_nodes]

        community_traces.append(go.Scatter(
            x=x_vals, y=y_vals, mode='markers+text',
            marker=dict(size=[max(10, min(30, f*2+8)) for f in freqs], 
                       color=colors[i], line=dict(width=2, color='white'),
                       opacity=0.85),
            text=[n[:15] + "…" if len(n) > 15 else n for n in comm_nodes],
            textposition="top center",
            textfont=dict(size=8, color=theme.get('font', '#000000')),
            hovertext=texts, hoverinfo='text',
            name=f"Community {i+1} ({len(comm_nodes)} nodes)"
        ))

    fig = go.Figure(data=[edge_trace] + community_traces)
    fig.update_layout(
        title=f"Community Detection (Modularity-based) — {len(communities)} communities",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
        font_color=theme.get('font', '#000000'),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=650,
        margin=dict(l=20, r=20, t=50, b=100)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show community summary
    comm_summary = []
    for i, comm in enumerate(communities):
        comm_concepts = list(comm)[:10]
        comm_summary.append({
            'Community': f"C{i+1}",
            'Size': len(comm),
            'Top Concepts': ", ".join(comm_concepts[:5])
        })
    st.dataframe(pd.DataFrame(comm_summary), use_container_width=True)


def render_concept_growth(df, valid_concepts, concept_abstract_map, all_texts, 
                          top_n=20, theme=None):
    """Render concept growth rate comparing early vs recent years."""
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if 'Year' not in df.columns or df['Year'].isna().all():
        st.info("Year data required for growth analysis.")
        return

    df_year = df.dropna(subset=['Year']).copy()
    df_year['Year'] = pd.to_numeric(df_year['Year'], errors='coerce')
    df_year = df_year.dropna(subset=['Year'])

    if len(df_year) < 10:
        st.info("Not enough year data for growth analysis.")
        return

    median_year = df_year['Year'].median()
    early_df = df_year[df_year['Year'] <= median_year]
    recent_df = df_year[df_year['Year'] > median_year]

    text_cols = [c for c in df_year.columns if any(k in c.lower() for k in ['abstract', 'title', 'summary', 'text'])]
    if not text_cols:
        st.info("No text columns for growth analysis.")
        return

    # Count concept occurrences in early vs recent
    top_concepts = sorted(valid_concepts, 
                          key=lambda c: len(concept_abstract_map.get(c, [])), 
                          reverse=True)[:top_n]

    growth_data = []
    for concept in top_concepts:
        early_count = 0
        for _, row in early_df.iterrows():
            text = " ".join([str(row.get(col, '')) for col in text_cols if pd.notna(row.get(col))])
            early_count += len(re.findall(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()))

        recent_count = 0
        for _, row in recent_df.iterrows():
            text = " ".join([str(row.get(col, '')) for col in text_cols if pd.notna(row.get(col))])
            recent_count += len(re.findall(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()))

        growth_rate = ((recent_count - early_count) / max(early_count, 1)) * 100 if early_count > 0 else 0
        growth_data.append({
            'Concept': concept,
            'Early Count': early_count,
            'Recent Count': recent_count,
            'Growth Rate (%)': round(growth_rate, 1)
        })

    growth_df = pd.DataFrame(growth_data).sort_values('Growth Rate (%)', ascending=True)

    fig = go.Figure()
    colors = ['#ef4444' if x < 0 else '#22c55e' for x in growth_df['Growth Rate (%)']]
    fig.add_trace(go.Bar(
        y=growth_df['Concept'].apply(lambda x: x[:25] + "…" if len(x) > 25 else x),
        x=growth_df['Growth Rate (%)'],
        orientation='h',
        marker_color=colors,
        text=growth_df['Growth Rate (%)'].apply(lambda x: f"{x:+.1f}%"),
        textposition='outside'
    ))
    fig.update_layout(
        title=f"Concept Growth Rate (Early ≤{int(median_year)} vs Recent >{int(median_year)})",
        xaxis_title="Growth Rate (%)",
        yaxis_title="",
        height=max(400, len(growth_df) * 25),
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
        font_color=theme.get('font', '#000000')
    )
    st.plotly_chart(fig, use_container_width=True)


def render_bubble_chart(nx_graph, valid_concepts, concept_abstract_map, distill_df, 
                        cmap_name="viridis", theme=None):
    """Render bubble chart: degree vs frequency, size = distillation efficiency."""
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 3:
        st.info("Not enough concepts for bubble chart.")
        return

    category_map = abstract_concepts_to_categories(valid_concepts)
    bubble_data = []
    for concept in valid_concepts:
        deg = nx_graph.degree(concept) if concept in nx_graph else 0
        freq = len(concept_abstract_map.get(concept, []))
        eff_row = distill_df[distill_df['concept'] == concept]
        efficiency = eff_row['distillation_efficiency'].values[0] if len(eff_row) > 0 else 0.5
        category = category_map.get(concept, 'general')
        bubble_data.append({
            'Concept': concept,
            'Degree': deg,
            'Frequency': freq,
            'Efficiency': efficiency,
            'Category': category
        })

    bubble_df = pd.DataFrame(bubble_data)

    fig = px.scatter(bubble_df, x='Degree', y='Frequency', size='Efficiency',
                     color='Category', hover_data=['Concept'],
                     title='Concept Importance Bubble Chart<br><sub>Size = Distillation Efficiency</sub>',
                     size_max=50,
                     color_discrete_sequence=get_colormap_colors(cmap_name, len(bubble_df['Category'].unique())))
    fig.update_layout(
        paper_bgcolor=theme.get('plotly_paper', '#ffffff'),
        plot_bgcolor=theme.get('plotly_bg', '#ffffff'),
        font_color=theme.get('font', '#000000'),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )
    st.plotly_chart(fig, use_container_width=True)


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

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
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
        st.markdown("- Defect-engineered Ag nanoparticles (ISF, EST, Twins)")
        st.markdown("- Technological applications (SERS, Catalysis, Sensing)")
        st.markdown("- Biomedical applications (Antibacterial, Theranostics)")
        st.markdown("- Synthesis & defect introduction (Plasma, irradiation, chemical reduction)")
        st.markdown("- Advanced characterization (HRTEM, STEM, XRD, UV-Vis, SERS mapping)")
        st.markdown("- Computational modeling (DFT, FDTD, DDA, MD)")

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

        
        st.markdown("---")

        # Graph Editing Section (only when analysis data exists)
        if st.session_state.get("analysis_data") is not None:
            data = st.session_state.analysis_data
            edits = render_graph_editing_sidebar(
                data["nx_graph"], data["valid_concepts"], 
                data["concept_to_id"], data["id_to_concept"],
                data["concept_abstract_map"], data.get("all_concepts", [])
            )
            if edits:
                if "pending_edits" not in st.session_state:
                    st.session_state.pending_edits = []
                st.session_state.pending_edits.extend(edits)
                st.success(f"Queued {len(edits)} edit(s). Click 'Apply Edits' below.")

            if st.button("✅ Apply All Edits", key="apply_all_edits", type="primary"):
                if st.session_state.get("pending_edits"):
                    with st.spinner("Applying edits..."):
                        nx_graph, valid_concepts, concept_to_id, id_to_concept, concept_abstract_map, modified =                             apply_graph_edits(
                                data["nx_graph"], data["valid_concepts"],
                                data["concept_to_id"], data["id_to_concept"],
                                data["concept_abstract_map"], data.get("all_concepts", []),
                                st.session_state.pending_edits
                            )
                        if modified:
                            st.session_state.analysis_data["nx_graph"] = nx_graph
                            st.session_state.analysis_data["valid_concepts"] = valid_concepts
                            st.session_state.analysis_data["concept_to_id"] = concept_to_id
                            st.session_state.analysis_data["id_to_concept"] = id_to_concept
                            st.session_state.analysis_data["concept_abstract_map"] = concept_abstract_map
                            st.session_state.pending_edits = []
                            st.success("Edits applied! Refreshing...")
                            st.rerun()
                        else:
                            st.info("No changes were made.")

            if st.button("🔄 Clear Edit Queue", key="clear_edits"):
                st.session_state.pending_edits = []
                st.success("Edit queue cleared")

            st.markdown("---")

            # Sunburst Options
            sunburst_cats, sunburst_branch = render_sunburst_sidebar(data["valid_concepts"])
            st.session_state['sunburst_categories'] = sunburst_cats
            st.session_state['sunburst_branchvalues'] = sunburst_branch
            st.markdown("---")

st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"🖥️ Device: {gpu_info}")

def main():
    st.title("🔬 Nanomaterials-ConceptGraph: Defect-Engineered Ag Nanoparticles Explorer")
    st.caption("Large-corpus concept graph builder for defect-engineered silver nanoparticles (ISF, EST, twins) and their technological applications (SERS, catalysis, sensing) • Optimized for defect characterization and functional performance")
    render_sidebar()
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = None
    if "pending_edits" not in st.session_state:
        st.session_state.pending_edits = []
    if "graph_filter" not in st.session_state:
        st.session_state.graph_filter = None

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
                    "all_concepts": all_concepts,
                    "config": config,
                    "df_filtered": df_filtered
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
        df_filtered = data.get("df_filtered", df_filtered)
        cmap = st.session_state.get('cmap_name', 'viridis')
        top_n_graph = st.session_state.get('top_n_graph', 200)
        viz_tab, distill_tab, scores_tab, valid_tab, export_tab, extra_viz_tab = st.tabs([
            "🎨 Visualization", "📊 Distillation", "🎯 Research Directions", "📐 Validation", "📥 Export", "✨ Extra Viz"
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
            theme = THEME_PRESETS.get(st.session_state.get('theme', 'Bright (Default)'), THEME_PRESETS["Bright (Default)"])

            top_n = st.session_state.get('top_n_graph', 0)

            # Apply graph filter if set
            if st.session_state.get('graph_filter'):
                filt = st.session_state['graph_filter']
                nx_graph = filter_graph_by_metrics(
                    nx_graph, concept_abstract_map,
                    min_degree=filt.get('min_degree', 0),
                    min_freq=filt.get('min_freq', 0),
                    max_degree=filt.get('max_degree'),
                    max_freq=filt.get('max_freq')
                )
                if st.button("🔄 Clear Filter", key="clear_filter"):
                    del st.session_state['graph_filter']
                    st.rerun()

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
                sunburst_cats = st.session_state.get('sunburst_categories', None)
                sunburst_branch = st.session_state.get('sunburst_branchvalues', 'total')
                labels, parents, values = build_category_hierarchy_filtered(
                    valid_concepts, concept_abstract_map,
                    selected_categories=sunburst_cats if sunburst_cats else None,
                    top_n_per_category=st.session_state.get('top_n_sunburst', 0)
                )
                render_sunburst_chart_enhanced(labels, parents, values, cmap_name=cmap, 
                                                branchvalues=sunburst_branch, theme=theme)

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

        # ─── EXTRA VISUALIZATIONS TAB ───
        with extra_viz_tab:
            st.subheader("✨ Innovative Visualizations")

            # Check if we have year data
            has_year_data = 'Year' in df_filtered.columns and not df_filtered['Year'].isna().all()

            ev_tab1, ev_tab2, ev_tab3, ev_tab4, ev_tab5, ev_tab6 = st.tabs([
                "📈 Timeline", "🔥 Heatmap", "🎯 t-SNE", "🌐 Communities", "📊 Growth", "🫧 Bubble"
            ])

            with ev_tab1:
                if has_year_data:
                    timeline_top_n = st.slider("Top N concepts for timeline", 5, 30, 15, key="timeline_top_n")
                    render_concept_timeline(df_filtered, valid_concepts, concept_abstract_map, 
                                            top_n=timeline_top_n, theme=theme)
                else:
                    st.info("📅 Concept Timeline requires a 'Year' column in your data.\n\n"
                            "Add publication year to your JSON records to enable this visualization.")

            with ev_tab2:
                heatmap_top_n = st.slider("Top N concepts for heatmap", 5, 50, 30, key="heatmap_top_n")
                render_cooccurrence_heatmap(nx_graph, valid_concepts, concept_abstract_map,
                                           top_n=heatmap_top_n, cmap_name=cmap, theme=theme)

            with ev_tab3:
                if st.button("🚀 Compute t-SNE Projection", key="btn_tsne"):
                    render_tsne_projection(valid_concepts, concept_abstract_map, 
                                          data.get("embed_model", load_embedding_model()),
                                          cmap_name=cmap, theme=theme)
                else:
                    st.info("Click the button to compute t-SNE projection (may take a moment).")

            with ev_tab4:
                if st.button("🚀 Detect Communities", key="btn_communities"):
                    render_community_detection(nx_graph, valid_concepts, concept_abstract_map,
                                              cmap_name=cmap, theme=theme)
                else:
                    st.info("Click the button to run community detection.")

            with ev_tab5:
                if has_year_data:
                    growth_top_n = st.slider("Top N concepts for growth", 5, 30, 20, key="growth_top_n")
                    render_concept_growth(df_filtered, valid_concepts, concept_abstract_map,
                                         data.get("all_texts", []), top_n=growth_top_n, theme=theme)
                else:
                    st.info("📊 Concept Growth requires a 'Year' column in your data.")

            with ev_tab6:
                render_bubble_chart(nx_graph, valid_concepts, concept_abstract_map, 
                                   distill_df, cmap_name=cmap, theme=theme)

if __name__ == "__main__":
    main()
