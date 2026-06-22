"""
Scopus CSV Format Converter & Unifier
=====================================
Converts malformed Scopus CSV exports (mixed delimiters, doubled quotes, 
trailing semicolons) to standard clean CSV format matching proper Scopus exports.

Output files are suffixed with:
  - _formatted.csv  : Clean CSV with proper comma-delimited format
  - _structured.json: JSON representation with metadata
"""

import streamlit as st
import pandas as pd
import json
import uuid
import re
import os
import io
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any, Literal


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_file_format(file_content: str) -> Literal["clean", "malformed", "unknown"]:
    """
    Detect whether a Scopus CSV file is clean (standard) or malformed.
    
    Clean:   comma-delimited, standard "quoted" fields
    Malformed: semicolon-heavy, doubled quotes "", trailing semicolons
    """
    first_line = file_content.split('\n')[0] if file_content else ""
    
    # Malformed indicators (strong signals)
    trailing_semicolons = len(first_line) - len(first_line.rstrip(';'))
    has_doubled_quotes = '""' in first_line
    semicolon_count = first_line.count(';')
    comma_in_quotes = first_line.count('","')
    
    # Heuristic: malformed files have tons of trailing semicolons OR
    # use "" quoting with ; delimiters
    if trailing_semicolons > 50:
        return "malformed"
    if has_doubled_quotes and semicolon_count > comma_in_quotes * 5:
        return "malformed"
    if first_line.startswith('"') and first_line.rstrip(';').endswith('"""'):
        return "malformed"
    
    # Clean file indicators
    if first_line.count('","') >= 20 and trailing_semicolons < 5:
        return "clean"
    
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# MALFORMED FILE PARSER (Category A/C)
# ─────────────────────────────────────────────────────────────────────────────

def parse_malformed_header(header_line: str) -> List[str]:
    """
    Extract column names from malformed header.
    
    Format: "Authors,""Author full names"",...""EID""";;;;;;;;;;;
    """
    # Strip trailing semicolons
    header_line = header_line.rstrip(';')
    
    # Extract content between outer quotes
    # Starts with " and ends with """ (the last field is ""EID"" wrapped by outer ")
    if not (header_line.startswith('"') and '"""' in header_line):
        raise ValueError("Malformed header: expected outer quotes and triple-quote ending")
    
    # Find the real content end (the """ after the last field)
    last_triple = header_line.rfind('"""')
    core = header_line[1:last_triple]  # Remove leading " and trailing """
    
    # Replace "" with " to normalize quoting
    core = core.replace('""', '"')
    
    # Now split by comma (the header uses comma separation)
    import csv
    from io import StringIO
    reader = csv.reader(StringIO(core), delimiter=',', quotechar='"')
    columns = list(reader)[0]
    
    return [c.strip() for c in columns if c.strip()]


def parse_malformed_data_row(line: str, expected_cols: int = 22) -> Optional[List[str]]:
    """
    Parse a single malformed data row into clean fields.
    
    Format: "Authors; ... ; Last Author,""Field2""; field3; ...; ""LastField""";;;;;;
    
    Strategy:
      1. Strip trailing semicolons
      2. Remove outer " wrapper (starts with ", ends with """)
      3. Authors field is everything before first ,"" 
      4. Rest is ;-delimited with "" quoting → convert to standard " quoting
      5. Parse rest with csv module
    """
    # Strip trailing semicolons
    line = line.rstrip(';')
    if not line:
        return None
    
    # Must start with " and contain """
    if not (line.startswith('"') and '"""' in line):
        # Fallback: try simpler format
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
            # Try to parse as semicolon-delimited
            import csv
            from io import StringIO
            reader = csv.reader(StringIO(line), delimiter=';', quotechar='"')
            return list(reader)[0]
        return None
    
    # Remove outer wrapper: leading " and trailing """
    # The trailing """ is: closing " of last field + "" (empty) + outer closing "
    # Actually it's: last field ends with "", then outer " closes → """
    core = line[1:-3]  # Remove leading " and trailing """
    
    # Find boundary: Authors field ends at first ,"" 
    # The pattern is: unquoted Authors text, then ,"" starts next field
    boundary = core.find(',""')
    
    if boundary == -1:
        # Try alternative: maybe it's just comma after Authors
        # Find first comma that's followed by quote-like pattern
        for i, c in enumerate(core):
            if c == ',' and i + 1 < len(core) and core[i+1] == '"':
                boundary = i
                break
    
    if boundary == -1:
        return None  # Can't find boundary
    
    # Extract Authors (unquoted, may contain ; and ,)
    authors = core[:boundary].strip()
    
    # Rest starts with "" (we skip the comma)
    rest = core[boundary + 1:]  # This starts with ""
    
    # Normalize: replace "" with standard "
    rest_cleaned = rest.replace('""', '"')
    
    # Parse as semicolon-delimited CSV with standard quoting
    import csv
    from io import StringIO
    reader = csv.reader(StringIO(rest_cleaned), delimiter=';', quotechar='"')
    parsed_rest = list(reader)[0]
    
    # Combine: Authors + rest of fields
    result = [authors] + parsed_rest
    
    # Pad or trim to expected column count
    if len(result) < expected_cols:
        result.extend([''] * (expected_cols - len(result)))
    elif len(result) > expected_cols:
        result = result[:expected_cols]
    
    return result


def parse_malformed_file(content: str) -> pd.DataFrame:
    """Parse entire malformed Scopus CSV file to DataFrame."""
    lines = content.strip().split('\n')
    if not lines:
        return pd.DataFrame()
    
    # Parse header
    columns = parse_malformed_header(lines[0])
    st.info(f"📄 Detected malformed format: {len(columns)} columns identified")
    
    # Parse data rows
    rows = []
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if not line:
            continue
        parsed = parse_malformed_data_row(line, expected_cols=len(columns))
        if parsed:
            rows.append(parsed)
        else:
            st.warning(f"⚠️ Could not parse row {i}, skipping")
    
    df = pd.DataFrame(rows, columns=columns)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN FILE PARSER (Category B) — Standard CSV
# ─────────────────────────────────────────────────────────────────────────────

def parse_clean_file(content: str) -> pd.DataFrame:
    """Parse standard comma-delimited Scopus CSV."""
    from io import StringIO
    df = pd.read_csv(StringIO(content), encoding='utf-8', dtype=str, keep_default_na=False)
    df = df.replace(r'^\s*$', None, regex=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_for_json(obj):
    """Recursively clean NaN, Inf, pandas NA for JSON output."""
    import math, numpy as np
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, np.generic):
        if np.issubdtype(type(obj), np.floating):
            val = float(obj)
            return None if (math.isnan(val) or math.isinf(val)) else val
        return obj.item()
    else:
        try:
            if pd.isna(obj) and obj is not None and not isinstance(obj, (str, bool, int, float)):
                return None
        except Exception:
            pass
        return obj


def dataframe_to_clean_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to clean, standard CSV string."""
    # Clean up: strip whitespace, normalize None
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else None)
    
    # Write to CSV with proper quoting
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8', quoting=1)  # QUOTE_ALL
    return output.getvalue()


def dataframe_to_json_records(df: pd.DataFrame, source_filename: str) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of JSON-ready records with metadata."""
    records = df.to_dict(orient='records')
    
    # Clean and add metadata
    clean_records = []
    for record in records:
        clean_record = {}
        for k, v in record.items():
            if pd.isna(v) or str(v).strip() == '':
                clean_record[k] = None
            else:
                clean_record[k] = str(v).strip()
        
        # Add metadata
        clean_record['_metadata'] = {
            'unique_id': str(uuid.uuid4()),
            'source_file': source_filename,
            'import_timestamp': datetime.now().isoformat(),
            'original_format': 'malformed' if 'malformed' in str(source_filename).lower() else 'clean'
        }
        clean_records.append(clean_record)
    
    # Final JSON sanitization
    return [sanitize_for_json(r) for r in clean_records]


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Scopus CSV Format Converter",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Scopus CSV Format Converter")
    st.markdown("""
    **Convert malformed Scopus CSV exports to clean, standard format.**
    
    | Input Format | Description | Output |
    |-------------|-------------|--------|
    | **Category A/C** (Malformed) | Mixed `;`/`,` delimiters, doubled quotes `""`, trailing `;` spam | Clean CSV + JSON |
    | **Category B** (Clean) | Standard comma-delimited CSV | Verified CSV + JSON |
    
    Output files are suffixed with `_formatted.csv` and `_structured.json`.
    """)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        output_format = st.radio(
            "Output format",
            ["CSV + JSON (both)", "CSV only", "JSON only"],
            index=0
        )
        
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload** your Scopus CSV files (A, B, or C format)
        2. The app **auto-detects** the format
        3. **Preview** the converted data
        4. **Download** `_formatted.csv` and/or `_structured.json`
        """)
    
    # File upload
    st.subheader("📤 Upload Scopus CSV Files")
    uploaded_files = st.file_uploader(
        "Select one or more CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload Category A (malformed), B (clean), or C (malformed) files"
    )
    
    if not uploaded_files:
        st.info("👆 Upload Scopus CSV files to begin conversion")
        
        # Example showing the difference
        with st.expander("📋 Format Examples"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Malformed (A/C)** — *will be converted*")
                st.code('''"Authors,""Author full names"",...""EID""";;;;;;;;;;;
"Yu M.; Huang R.;...,""Yu, Meng..."";...;""Title""";;;;;''', language='text')
            with col2:
                st.markdown("**Clean (B)** — *passes through*")
                st.code('''"Authors","Author full names",...,"EID"
"Wei H.-S.;...","Wei, Hsiang...",...''', language='text')
        return
    
    # Process each file
    all_results = []
    
    for uploaded_file in uploaded_files:
        st.divider()
        filename = uploaded_file.name
        base_name = Path(filename).stem
        
        st.subheader(f"📄 {filename}")
        
        # Read content
        try:
            content = uploaded_file.read().decode('utf-8-sig')
        except UnicodeDecodeError:
            try:
                content = uploaded_file.read().decode('latin-1')
            except Exception as e:
                st.error(f"❌ Cannot decode {filename}: {e}")
                continue
        
        # Detect format
        fmt = detect_file_format(content)
        
        if fmt == "malformed":
            st.info("🔍 **Format detected:** Malformed (Category A/C style)")
            with st.spinner("Converting malformed CSV to clean format..."):
                try:
                    df = parse_malformed_file(content)
                except Exception as e:
                    st.error(f"❌ Conversion failed: {e}")
                    continue
        elif fmt == "clean":
            st.info("🔍 **Format detected:** Clean (Category B style)")
            try:
                df = parse_clean_file(content)
            except Exception as e:
                st.error(f"❌ Parse failed: {e}")
                continue
        else:
            st.warning("⚠️ **Format uncertain** — attempting clean parse...")
            try:
                df = parse_clean_file(content)
            except Exception:
                try:
                    df = parse_malformed_file(content)
                except Exception as e:
                    st.error(f"❌ All parsing attempts failed: {e}")
                    continue
        
        # Show preview
        st.success(f"✅ Parsed **{len(df)} rows** × **{len(df.columns)} columns**")
        
        with st.expander("🔍 Preview first 3 rows", expanded=False):
            st.dataframe(df.head(3), use_container_width=True)
        
        # Store result
        all_results.append({
            'filename': filename,
            'base_name': base_name,
            'dataframe': df,
            'format_detected': fmt
        })
    
    # Download section
    if all_results:
        st.divider()
        st.header("💾 Download Converted Files")
        
        for result in all_results:
            base = result['base_name']
            df = result['dataframe']
            fmt = result['format_detected']
            
            st.subheader(f"📦 {result['filename']}")
            st.caption(f"Detected as: **{fmt}** | Rows: {len(df)} | Columns: {len(df.columns)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_data = dataframe_to_clean_csv(df)
                csv_filename = f"{base}_formatted.csv"
                st.download_button(
                    label=f"⬇️ Download `{csv_filename}`",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"Clean comma-delimited CSV ({len(csv_data):,} bytes)")
            
            with col2:
                # JSON download
                records = dataframe_to_json_records(df, result['filename'])
                json_data = json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False)
                json_filename = f"{base}_structured.json"
                st.download_button(
                    label=f"⬇️ Download `{json_filename}`",
                    data=json_data,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
                st.caption(f"Structured JSON with metadata ({len(json_data):,} bytes)")
            
            # Show column mapping
            with st.expander("📊 Column mapping", expanded=False):
                st.markdown("**Columns in output:**")
                for i, col in enumerate(df.columns, 1):
                    st.text(f"  {i:2d}. {col}")
    
    # Combined download if multiple files
    if len(all_results) > 1:
        st.divider()
        st.header("🔗 Combined Output")
        
        combined_df = pd.concat([r['dataframe'] for r in all_results], ignore_index=True)
        st.info(f"Combined total: **{len(combined_df)} rows** from **{len(all_results)} files**")
        
        col1, col2 = st.columns(2)
        with col1:
            combined_csv = dataframe_to_clean_csv(combined_df)
            st.download_button(
                label="⬇️ Download Combined CSV",
                data=combined_csv,
                file_name=f"scopus_combined_formatted.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            combined_records = []
            for r in all_results:
                combined_records.extend(dataframe_to_json_records(r['dataframe'], r['filename']))
            combined_json = json.dumps(combined_records, indent=2, ensure_ascii=False, allow_nan=False)
            st.download_button(
                label="⬇️ Download Combined JSON",
                data=combined_json,
                file_name=f"scopus_combined_structured.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
