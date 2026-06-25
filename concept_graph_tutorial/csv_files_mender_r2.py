"""
Scopus CSV Format Learner & Converter
=====================================
1. User uploads CORRECT format file (Category B) — app learns the structure
2. User uploads INCORRECT format file(s) (Category A/C) — app converts to match learned structure
"""

import streamlit as st
import pandas as pd
import csv
import io
from pathlib import Path
from typing import List, Dict, Literal

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING: FIX MALFORMED CSV
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_malformed_csv(content: str) -> str:
    """
    Safely detects and fixes CSV files where each entire row is wrapped in an extra 
    layer of double quotes, with optional trailing semicolons.
    """
    if not content.strip():
        return content

    # Quick check on the first line to avoid unnecessary processing of normal files
    first_line = content.split('\n')[0].strip()
    try:
        reader = csv.reader(io.StringIO(first_line))
        first_row = next(reader)
        # If the first line parses as a single field but contains commas, 
        # it's the malformed format (entire line wrapped in quotes).
        if not (len(first_row) == 1 and ',' in first_row[0]):
            return content 
    except Exception:
        return content

    # Parse the whole content safely using csv.reader
    # This naturally handles multiline fields (like long abstracts) and escaped quotes
    reader = csv.reader(io.StringIO(content))
    fixed_lines = []
    
    for row in reader:
        if not row:
            fixed_lines.append("")
            continue
            
        inner_str = row[0]
        
        # Strip trailing semicolons that were appended outside the inner quotes
        inner_str = inner_str.rstrip(';')
        
        fixed_lines.append(inner_str)
        
    return '\n'.join(fixed_lines)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: LEARN FROM CORRECT FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def learn_from_correct_format(content: str) -> Dict:
    """Parse correct format and extract structural information."""
    df = pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False)
    
    structure = {
        'columns': list(df.columns),
        'num_columns': len(df.columns),
        'sample_row': df.iloc[0].to_dict() if len(df) > 0 else {},
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
    }
    
    for col in ['Authors', 'Author full names', 'Author(s) ID', 'Title', 'Year', 'EID']:
        if col in df.columns and len(df) > 0:
            sample = str(df[col].iloc[0])
            structure[f'{col}_pattern'] = {
                'has_semicolon': ';' in sample,
                'has_comma': ',' in sample,
                'has_quote': '"' in sample,
                'length': len(sample),
            }
            
    return structure

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: DETECT AND PARSE INCORRECT FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def detect_format_type(content: str) -> Literal['correct', 'incorrect', 'unknown']:
    """Detect if file is correct or incorrect Scopus CSV format."""
    if not content.strip():
        return 'unknown'
        
    first_line = content.split('\n')[0].strip()
    
    try:
        reader = csv.reader(io.StringIO(first_line))
        first_row = next(reader)
        # Incorrect format: entire line is wrapped in quotes, parsing as 1 field with commas
        if len(first_row) == 1 and ',' in first_row[0]:
            return 'incorrect'
    except Exception:
        pass
        
    # Correct format indicators
    if first_line.count('","') >= 5 and not first_line.rstrip().endswith(';'):
        return 'correct'
    
    return 'unknown'

def parse_incorrect_format(content: str, learned_structure: Dict) -> pd.DataFrame:
    """
    Parse incorrect Scopus CSV format by transforming it to correct format.
    """
    # Step 1: Clean the malformed CSV string
    cleaned_content = preprocess_malformed_csv(content)
    
    # Step 2: Parse the cleaned content using pandas
    try:
        df = pd.read_csv(io.StringIO(cleaned_content), dtype=str, keep_default_na=False)
    except Exception as e:
        st.error(f"Failed to parse cleaned CSV: {e}")
        return pd.DataFrame(columns=learned_structure['columns'])
    
    # Step 3: Align with learned structure
    expected_cols = learned_structure['columns']
    
    # Add missing columns as empty strings
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ''
            
    # Keep only the expected columns in the correct order
    df = df[[col for col in expected_cols if col in df.columns]]
    
    # Reorder to exactly match learned structure
    df = df[expected_cols]
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_as_correct_csv(df: pd.DataFrame) -> str:
    """Format DataFrame as correct Scopus CSV."""
    output = io.StringIO()
    df.to_csv(output, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    return output.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Scopus CSV Format Learner", page_icon="🎓", layout="wide")
    
    st.title("🎓 Scopus CSV Format Learner & Converter")
    st.markdown("""
    **Two-phase workflow:**
    1. **Learn**: Upload a correctly-formatted Scopus CSV (Category B)
    2. **Convert**: Upload incorrectly-formatted files (Category A/C) to convert them
    """)
    
    # Initialize session state
    if 'learned_structure' not in st.session_state:
        st.session_state.learned_structure = None
    
    # ── Phase 1: Learn ──────────────────────────────────────────────────────
    st.header("Phase 1: Learn Correct Format")
    
    correct_file = st.file_uploader(
        "Upload a CORRECTLY formatted Scopus CSV (Category B)",
        type=['csv'],
        key='correct_uploader'
    )
    
    if correct_file:
        # Handle potential encoding issues gracefully
        try:
            content = correct_file.read().decode('utf-8-sig')
        except UnicodeDecodeError:
            content = correct_file.read().decode('latin-1')
            
        fmt = detect_format_type(content)
        
        if fmt == 'incorrect':
            st.error("❌ This file appears to be in INCORRECT format. Please upload a correct one.")
        else:
            try:
                structure = learn_from_correct_format(content)
                st.session_state.learned_structure = structure
                
                st.success(f"✅ Learned structure: **{structure['num_columns']} columns**")
                
                with st.expander("View learned structure"):
                    st.json({
                        'columns': structure['columns'],
                        'num_columns': structure['num_columns'],
                        'sample_patterns': {k: v for k, v in structure.items() if '_pattern' in k}
                    })
            except Exception as e:
                st.error(f"❌ Failed to learn: {e}")
    
    # ── Phase 2: Convert ────────────────────────────────────────────────────
    st.header("Phase 2: Convert Incorrect Format")
    
    if st.session_state.learned_structure is None:
        st.info("⬆️ Please upload a correct format file first to enable conversion.")
        return
    
    incorrect_files = st.file_uploader(
        "Upload INCORRECTLY formatted Scopus CSV files (Category A/C)",
        type=['csv'],
        accept_multiple_files=True,
        key='incorrect_uploader'
    )
    
    if not incorrect_files:
        st.info("⬆️ Upload incorrect format files to convert")
        return
    
    for uploaded_file in incorrect_files:
        st.divider()
        st.subheader(f"📄 {uploaded_file.name}")
        
        try:
            content = uploaded_file.read().decode('utf-8-sig')
        except UnicodeDecodeError:
            content = uploaded_file.read().decode('latin-1')
            
        fmt = detect_format_type(content)
        
        if fmt == 'correct':
            st.warning("⚠️ This file appears to already be in correct format. Skipping conversion.")
            continue
        
        if fmt == 'unknown':
            st.warning("⚠️ Could not definitively detect format. Attempting conversion anyway.")
        
        st.info("🔍 Detected: **Incorrect format** — converting...")
        
        try:
            with st.spinner("Converting..."):
                df = parse_incorrect_format(content, st.session_state.learned_structure)
            
            if df.empty:
                st.warning("⚠️ Conversion resulted in an empty DataFrame. Check file contents.")
                continue

            st.success(f"✅ Converted: **{len(df)} rows** × **{len(df.columns)} columns**")
            
            with st.expander("Preview"):
                st.dataframe(df.head(3), use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            base_name = Path(uploaded_file.name).stem
            
            with col1:
                csv_output = format_as_correct_csv(df)
                st.download_button(
                    label=f"⬇️ Download `{base_name}_formatted.csv`",
                    data=csv_output,
                    file_name=f"{base_name}_formatted.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_output = df.to_json(orient='records', indent=2, force_ascii=False)
                st.download_button(
                    label=f"⬇️ Download `{base_name}_structured.json`",
                    data=json_output,
                    file_name=f"{base_name}_structured.json",
                    mime="application/json",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Conversion failed: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
