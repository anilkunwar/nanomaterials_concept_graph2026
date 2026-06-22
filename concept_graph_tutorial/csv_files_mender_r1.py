"""
Scopus CSV Format Learner & Converter
=====================================
1. User uploads CORRECT format file (Category B) — app learns the structure
2. User uploads INCORRECT format file(s) (Category A/C) — app converts to match learned structure
"""

import streamlit as st
import pandas as pd
import csv
import re
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: LEARN FROM CORRECT FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def learn_from_correct_format(content: str) -> Dict:
    """Parse correct format and extract structural information."""
    # Parse with standard CSV
    df = pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False)
    
    # Learn column names and sample data patterns
    structure = {
        'columns': list(df.columns),
        'num_columns': len(df.columns),
        'sample_row': df.iloc[0].to_dict() if len(df) > 0 else {},
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
    }
    
    # Learn patterns for key fields
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
    first_line = content.split('\n')[0] if content else ''
    
    # Incorrect format indicators (strong)
    if first_line.endswith(';' * 50):  # Trailing semicolons
        return 'incorrect'
    if first_line.count('""') > 20:  # Doubled quotes
        return 'incorrect'
    if first_line.startswith('"') and '"""' in first_line:  # Outer wrapper + triple quote
        return 'incorrect'
    
    # Correct format indicators
    if first_line.count('","') >= 20 and not first_line.endswith(';'):
        return 'correct'
    
    return 'unknown'


def parse_incorrect_format(content: str, learned_structure: Dict) -> pd.DataFrame:
    """
    Parse incorrect Scopus CSV format by transforming it to correct format.
    
    The incorrect format has these characteristics:
    - Outer " wrapper around each row
    - Inner "" quotes for fields containing commas
    - ; as field separator (but inconsistently mixed with ,)
    - Trailing ; spam
    - Authors field is unquoted and contains ; as data
    """
    lines = content.strip().split('\n')
    if not lines:
        return pd.DataFrame(columns=learned_structure['columns'])
    
    expected_cols = learned_structure['num_columns']
    
    # Parse header
    header_cols = parse_incorrect_header(lines[0])
    if len(header_cols) != expected_cols:
        st.warning(f"Header column count mismatch: found {len(header_cols)}, expected {expected_cols}")
    
    # Parse data rows
    rows = []
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if not line:
            continue
        
        parsed = parse_incorrect_data_row(line, expected_cols)
        if parsed:
            rows.append(parsed)
        else:
            st.warning(f"Could not parse row {i}")
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=learned_structure['columns'][:len(rows[0]) if rows else 0])
    
    # Ensure all expected columns exist
    for col in learned_structure['columns']:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder to match learned structure
    df = df[learned_structure['columns']]
    
    return df


def parse_incorrect_header(header_line: str) -> List[str]:
    """Extract column names from incorrect format header."""
    # Strip trailing semicolons
    header_line = header_line.rstrip(';')
    
    # Remove outer " and trailing """
    if header_line.startswith('"') and header_line.endswith('"""'):
        core = header_line[1:-3]
    else:
        core = header_line
    
    # Replace "" with "
    core = core.replace('""', '"')
    
    # Split by comma
    cols = []
    reader = csv.reader(io.StringIO(core), delimiter=',', quotechar='"')
    for row in reader:
        cols = [c.strip() for c in row if c.strip()]
        break
    
    return cols


def parse_incorrect_data_row(line: str, expected_cols: int) -> Optional[List[str]]:
    """
    Parse a single incorrect data row.
    
    Strategy:
    1. Remove outer wrapper and trailing semicolons
    2. Find Authors field boundary (first occurrence of ,"")
    3. Authors = everything before boundary
    4. Rest = everything after boundary, parsed as mixed-delimiter CSV
    5. Combine and map to expected columns
    """
    # Strip trailing semicolons
    line = line.rstrip(';')
    if not line:
        return None
    
    # Remove outer " and trailing """
    if not (line.startswith('"') and line.endswith('"""')):
        # Try alternative: just outer quotes
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
    else:
        line = line[1:-3]
    
    # Find boundary between Authors and rest
    # The boundary is the first occurrence of ,"" 
    boundary = line.find(',""')
    if boundary == -1:
        # Try with ,"
        boundary = line.find(',"')
    
    if boundary == -1:
        return None
    
    authors = line[:boundary].strip()
    rest = line[boundary + 1:]  # Starts with "" or "
    
    # Replace "" with " to normalize
    rest = rest.replace('""', '"')
    
    # Now parse rest. The challenge: mixed ; and , delimiters
    # Strategy: use ; as primary delimiter, but handle quoted fields with commas
    
    # First, let's try to identify field boundaries
    # A field is either:
    # - "quoted content" (may contain ; and ,)
    # - bare content (no ; or ,)
    # Separated by ; or ,
    
    fields = parse_mixed_delimiter_rest(rest)
    
    # Combine Authors with rest
    result = [authors] + fields
    
    # Map to expected columns
    # The incorrect format expands multi-value fields (Author full names, Author IDs)
    # into separate physical fields. We need to combine them back.
    
    result = combine_expanded_fields(result, expected_cols)
    
    # Pad or trim
    if len(result) < expected_cols:
        result.extend([''] * (expected_cols - len(result)))
    elif len(result) > expected_cols:
        result = result[:expected_cols]
    
    return result


def parse_mixed_delimiter_rest(rest: str) -> List[str]:
    """Parse the rest of the row after Authors field."""
    fields = []
    current = []
    in_quotes = False
    i = 0
    
    while i < len(rest):
        char = rest[i]
        next_char = rest[i+1] if i+1 < len(rest) else ''
        
        if char == '"':
            if in_quotes:
                if next_char == '"':
                    # Escaped quote
                    current.append('"')
                    i += 2
                else:
                    # End of quoted field
                    in_quotes = False
                    i += 1
            else:
                # Start of quoted field
                in_quotes = True
                i += 1
        elif (char == ';' or char == ',') and not in_quotes:
            # Field separator
            field = ''.join(current).strip()
            if field or fields:  # Only add if we have content or previous fields
                fields.append(field)
            current = []
            i += 1
        else:
            current.append(char)
            i += 1
    
    # Add last field
    if current or fields:
        fields.append(''.join(current).strip())
    
    return fields


def combine_expanded_fields(fields: List[str], expected_cols: int) -> List[str]:
    """
    Combine expanded fields back into logical columns.
    
    The incorrect format splits multi-value fields (Author full names, Author IDs)
    into separate physical fields. We need to detect and combine them.
    """
    if len(fields) <= expected_cols:
        return fields
    
    # Heuristic: the first 3 fields are Authors, Author full names, Author IDs
    # These may be expanded in the incorrect format
    
    # Authors is already field 0 (passed separately)
    # fields[0:] are the rest
    
    # Try to identify where the expansion happened
    # Author full names typically contain patterns like "Name, Name (ID)"
    # Author IDs are typically numeric
    
    result = [fields[0]]  # Authors
    
    # Find Author full names fields (contain names with commas and parentheses)
    # and Author ID fields (numeric)
    
    # Heuristic: group consecutive fields that look like author names
    author_name_fields = []
    author_id_fields = []
    remaining_start = 1
    
    for i, field in enumerate(fields[1:], 1):
        # Check if field looks like an author name: "Last, First (ID)" or "Last, First"
        if re.match(r'^[A-Z][a-zA-Z\\-\\s]+,\\s*[A-Z][a-zA-Z\\-\\s]+(\\s*\\(\\d+\\))?$', field):
            author_name_fields.append(field)
            remaining_start = i + 1
        # Check if field looks like an author ID: numeric
        elif field.isdigit() and len(field) > 5:
            author_id_fields.append(field)
            remaining_start = i + 1
        else:
            break
    
    # Combine author names
    if author_name_fields:
        result.append('; '.join(author_name_fields))
    else:
        result.append('')
    
    # Combine author IDs
    if author_id_fields:
        result.append('; '.join(author_id_fields))
    else:
        result.append('')
    
    # Add remaining fields
    result.extend(fields[remaining_start:])
    
    return result


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
        content = correct_file.read().decode('utf-8-sig')
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
        
        content = uploaded_file.read().decode('utf-8-sig')
        fmt = detect_format_type(content)
        
        if fmt == 'correct':
            st.warning("⚠️ This file appears to already be in correct format. Skipping conversion.")
            continue
        
        if fmt == 'unknown':
            st.warning("⚠️ Could not detect format. Attempting conversion anyway.")
        
        st.info("🔍 Detected: **Incorrect format** — converting...")
        
        try:
            with st.spinner("Converting..."):
                df = parse_incorrect_format(content, st.session_state.learned_structure)
            
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
