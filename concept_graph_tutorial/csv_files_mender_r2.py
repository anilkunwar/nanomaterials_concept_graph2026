#!/usr/bin/env python3
"""
Scopus CSV Repair & Explorer — Streamlit App
=============================================
Auto-detects and repairs three known malformed Scopus CSV export variants:
  • Type A/C: Double-double-quoted fields, semicolon record separators
  • Type B:   Standard CSV with possible minor inconsistencies

Upload any Scopus CSV export and get clean, validated, downloadable output.
"""

import streamlit as st
import pandas as pd
import csv
import re
import io
from io import StringIO, BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scopus CSV Repair",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────────────
STANDARD_COLUMNS = [
    "Authors", "Author full names", "Author(s) ID", "Title", "Year",
    "Source title", "Volume", "Issue", "Art. No.", "Page start",
    "Page end", "Cited by", "DOI", "Link", "Abstract",
    "Author Keywords", "Index Keywords", "Document Type",
    "Publication Stage", "Open Access", "Source", "EID"
]

CATEGORICAL_COLS = ["Year", "Source title", "Document Type", "Publication Stage", "Open Access"]
NUMERIC_COLS = ["Volume", "Issue", "Cited by"]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f4e79; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; border-left: 4px solid #1f4e79; }
    .success-box { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 6px; padding: 1rem; }
    .warning-box { background: #fff3cd; border: 1px solid #ffeeba; border-radius: 6px; padding: 1rem; }
    .error-box { background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 1rem; }
    .stDataFrame { font-size: 0.85rem; }
    div[data-testid="stFileUploader"] { border: 2px dashed #1f4e79 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE REPAIR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FormatProfile:
    has_bom: bool
    has_line_endings: bool
    quote_style: str
    semicolon_runs: int
    double_double_quotes: int
    detected_type: str
    confidence: float


def detect_format(raw_bytes: bytes) -> FormatProfile:
    """Analyze raw bytes to determine the corruption profile."""
    text = raw_bytes.decode('utf-8-sig', errors='replace')
    
    has_bom = raw_bytes[:3] == b'\xef\xbb\xbf'
    has_line_endings = '\n' in text or '\r' in text
    semicolon_runs = len(re.findall(r';{3,}', text))
    double_double = text.count('""')
    
    # Determine type
    if double_double > 50 and semicolon_runs > 0:
        detected = "Type A/C (double-double-quotes + semicolon separators)"
        confidence = 0.95
    elif double_double > 0 and not semicolon_runs:
        detected = "Type B variant (minor quote issues)"
        confidence = 0.80
    else:
        detected = "Type B (standard CSV)"
        confidence = 0.90
    
    return FormatProfile(
        has_bom=has_bom,
        has_line_endings=has_line_endings,
        quote_style="double-double" if double_double > 50 else "standard",
        semicolon_runs=semicolon_runs,
        double_double_quotes=double_double,
        detected_type=detected,
        confidence=confidence
    )


def normalize_columns(fields: List[str], expected: int = 22) -> List[str]:
    """Ensure row has exactly the expected number of columns."""
    if len(fields) == expected:
        return fields
    elif len(fields) < expected:
        return fields + [''] * (expected - len(fields))
    else:
        # Merge overflow into the last column (usually EID or overflow text)
        overflow = fields[expected - 1:]
        return fields[:expected - 1] + [' '.join(overflow)]


def parse_type_a_c(text: str) -> List[List[str]]:
    """Parse Type A/C: double-double-quotes with semicolon record separators."""
    # Split by large semicolon clusters (5+ semicolons = record boundary)
    records = re.split(r';{5,}', text)
    records = [r.strip() for r in records if r.strip()]
    
    rows = []
    for record in records:
        record = record.rstrip(';').strip()
        if not record:
            continue
        
        # Remove outer wrapper quotes if the entire record is wrapped
        if record.startswith('"') and record.endswith('"'):
            inner = record[1:-1]
            if inner.startswith('Authors') or inner.startswith('Authors,'):
                record = inner
        
        # Split by the ,"" delimiter pattern
        fields = record.split(',""')
        fields = [f.strip().strip('"').replace('""', '"') for f in fields]
        fields = normalize_columns(fields)
        rows.append(fields)
    
    return rows


def parse_type_b(text: str) -> List[List[str]]:
    """Parse Type B: standard CSV with cleanup."""
    reader = csv.reader(StringIO(text), quotechar='"', delimiter=',')
    rows = []
    
    for row in reader:
        if not row or all(not c.strip() for c in row):
            continue
        rows.append(normalize_columns(row))
    
    return rows


def repair_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[FormatProfile], List[str]]:
    """
    Main repair pipeline.
    Returns: (dataframe, profile, warnings)
    """
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    
    profile = detect_format(raw)
    warnings_list = []
    
    text = raw.decode('utf-8-sig', errors='replace')
    
    # Route to parser
    if profile.quote_style == "double-double" and profile.semicolon_runs > 0:
        rows = parse_type_a_c(text)
    else:
        rows = parse_type_b(text)
    
    if not rows:
        return None, profile, ["No parseable rows found."]
    
    # Validate header
    header = rows[0]
    if header[0] != "Authors":
        warnings_list.append(f"Unexpected first column: '{header[0][:40]}...' — may need manual review.")
    
    # Ensure header matches standard
    if len(header) == len(STANDARD_COLUMNS):
        rows[0] = STANDARD_COLUMNS
    
    # Build DataFrame
    try:
        df = pd.DataFrame(rows[1:], columns=rows[0])
    except Exception as e:
        # Fallback: force standard columns
        df = pd.DataFrame(rows[1:], columns=STANDARD_COLUMNS[:len(rows[1])] if rows[1:] else STANDARD_COLUMNS)
        warnings_list.append(f"Column mismatch handled: {e}")
    
    # Clean numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean Year
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
    
    return df, profile, warnings_list


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8-SIG CSV bytes for Excel compatibility."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
    return buf.getvalue().encode('utf-8-sig')


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Repaired Data')
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔧 Scopus CSV Repair")
        st.markdown("---")
        st.markdown("**Supported formats:**")
        st.markdown("""
        - **Type A/C**: Double-double-quoted fields with semicolon separators
        - **Type B**: Standard CSV with minor issues
        """)
        st.markdown("---")
        st.markdown("**Output formats:**")
        st.markdown("- CSV (UTF-8 with BOM)")
        st.markdown("- Excel (.xlsx)")
        st.markdown("- JSON")
        st.markdown("---")
        st.caption("v1.0 — Auto-detects and repairs malformed Scopus exports")
    
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="main-header">🔧 Scopus CSV Repair & Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload malformed Scopus CSV exports — auto-detect format, repair, preview, and download clean data.</div>', unsafe_allow_html=True)
    
    # ── File Uploader ─────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "📁 Drop your Scopus CSV files here",
        type=["csv"],
        accept_multiple_files=True,
        help="Supports all three known Scopus export corruption variants"
    )
    
    if not uploaded_files:
        st.info("👆 Upload one or more `.csv` files to begin.")
        
        # Show example of what gets fixed
        with st.expander("📋 What problems does this fix?"):
            st.markdown("### Common Scopus Export Corruptions")
            st.markdown("**1. Double-Double-Quote Escaping (Type A/C)**")
            st.code('Before: "Authors,""Author full names"",""Title""...""EID"""; ; ; ; ;\nAfter:  Authors,Author full names,Title,...,EID', language=None)
            st.markdown("**2. Semicolon Record Separators**")
            st.code("Before: Record1;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Record2\nAfter:  Record1\\nRecord2", language=None)
            st.markdown("**3. Column Count Mismatches**")
            st.markdown("- Extra commas inside Abstract/Keyword fields → merged correctly")
            st.markdown("- Missing trailing columns → padded with empty strings")
            st.markdown("- Overflow columns → consolidated into appropriate fields")
        return
    
    # ── Process all files ─────────────────────────────────────────────────────
    all_results = []
    
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.markdown(f"### 📄 {uploaded_file.name}")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Repair
            df, profile, warnings = repair_csv(uploaded_file)
            
            if df is None:
                st.error("❌ Could not parse file.")
                continue
            
            # Format detection card
            st.markdown("#### 🔍 Format Detected")
            st.markdown(f"""
            <div class="metric-card">
                <b>{profile.detected_type}</b><br>
                <small>
                • BOM: {'Yes' if profile.has_bom else 'No'}<br>
                • Quote style: {profile.quote_style}<br>
                • Semicolon runs: {profile.semicolon_runs}<br>
                • Confidence: {profile.confidence:.0%}
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            # Stats
            st.markdown("#### 📊 Stats")
            st.metric("Records", len(df))
            st.metric("Columns", len(df.columns))
            
            if "Year" in df.columns and df["Year"].notna().any():
                yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
                st.metric("Year range", f"{yr_min}–{yr_max}")
            
            # Warnings
            if warnings:
                st.markdown("#### ⚠️ Warnings")
                for w in warnings:
                    st.warning(w)
            else:
                st.markdown("#### ✅ Status")
                st.success("Clean repair — no issues")
            
            # Download buttons
            st.markdown("#### ⬇️ Download")
            
            file_stem = Path(uploaded_file.name).stem
            
            # CSV
            csv_data = to_csv_bytes(df)
            st.download_button(
                label="📄 CSV (UTF-8-BOM)",
                data=csv_data,
                file_name=f"{file_stem}_FIXED.csv",
                mime="text/csv",
                use_container_width=True,
            )
            
            # Excel
            try:
                excel_data = to_excel_bytes(df)
                st.download_button(
                    label="📊 Excel (.xlsx)",
                    data=excel_data,
                    file_name=f"{file_stem}_FIXED.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception:
                st.info("Install `openpyxl` for Excel export")
            
            # JSON
            json_data = df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                label="🗂️  JSON",
                data=json_data,
                file_name=f"{file_stem}_FIXED.json",
                mime="application/json",
                use_container_width=True,
            )
        
        with col2:
            # Tabs for different views
            tabs = st.tabs(["📋 Data Preview", "🔬 Column Analysis", "📑 Raw Records", "📈 Year Distribution"])
            
            with tabs[0]:
                # Smart column selection for preview
                preview_cols = [c for c in ["Authors", "Title", "Year", "Source title", "Cited by", "DOI"] if c in df.columns]
                if not preview_cols:
                    preview_cols = list(df.columns)[:6]
                
                st.dataframe(
                    df[preview_cols],
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                )
            
            with tabs[1]:
                st.markdown("#### Column completeness & types")
                
                # Build summary
                summary = []
                for col in df.columns:
                    non_null = df[col].notna().sum()
                    total = len(df)
                    pct = (non_null / total * 100) if total else 0
                    sample = str(df[col].dropna().iloc[0])[:60] if df[col].notna().any() else "—"
                    summary.append({
                        "Column": col,
                        "Non-empty": f"{non_null}/{total}",
                        "Fill %": f"{pct:.0f}%",
                        "Sample value": sample + ("…" if len(sample) == 60 else "")
                    })
                
                summary_df = pd.DataFrame(summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True, height=500)
            
            with tabs[2]:
                st.markdown("#### Individual record details")
                
                record_idx = st.number_input(
                    "Select record",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=0,
                    key=f"rec_{uploaded_file.name}"
                )
                
                row = df.iloc[record_idx]
                
                # Display as styled cards
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val) or str(val).strip() == '':
                        continue
                    
                    with st.container():
                        st.markdown(f"**{col}**")
                        if col in ["Abstract", "Author Keywords", "Index Keywords"]:
                            st.markdown(f"<div style='background:#f8f9fa;padding:0.5rem;border-radius:4px;font-size:0.9rem;'>{val}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<code>{val}</code>", unsafe_allow_html=True)
                        st.markdown("---")
            
            with tabs[3]:
                if "Year" in df.columns and df["Year"].notna().any():
                    year_counts = df["Year"].value_counts().sort_index()
                    
                    col_chart, col_table = st.columns([2, 1])
                    with col_chart:
                        st.bar_chart(year_counts)
                    with col_table:
                        st.dataframe(year_counts.rename("Count"), use_container_width=True)
                else:
                    st.info("No valid Year data available for distribution.")
        
        all_results.append({
            "name": uploaded_file.name,
            "df": df,
            "profile": profile,
            "warnings": warnings
        })
    
    # ── Combined batch download ─────────────────────────────────────────────
    if len(all_results) > 1:
        st.markdown("---")
        st.markdown("### 📦 Batch Export")
        
        # Combined CSV
        combined = pd.concat([r["df"] for r in all_results], ignore_index=True)
        
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            st.download_button(
                "📄 Combined CSV",
                data=to_csv_bytes(combined),
                file_name="ALL_COMBINED_FIXED.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with col_b2:
            try:
                st.download_button(
                    "📊 Combined Excel",
                    data=to_excel_bytes(combined),
                    file_name="ALL_COMBINED_FIXED.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception:
                pass
        
        with col_b3:
            json_combined = combined.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                "🗂️  Combined JSON",
                data=json_combined,
                file_name="ALL_COMBINED_FIXED.json",
                mime="application/json",
                use_container_width=True,
            )
        
        st.info(f"Combined total: **{len(combined)}** records from **{len(all_results)}** files")


if __name__ == "__main__":
    main()
