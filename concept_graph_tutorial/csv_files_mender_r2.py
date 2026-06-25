#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Scopus CSV Format Learner & Converter — Bulletproof Edition v6
===============================================================

THE CORRUPTION PATTERN (observed in Scopus Category A CSV exports):

Correct (Category B):  "Authors","Author full names","Title",...,"EID"
Incorrect (Category A): "Authors,""Author full names"",""Title"",...,""EID""";;;;;;;;;;

The corruption is:
1. Entire row wrapped in extra outer quotes: "...content..."
2. All internal quotes doubled: " -> ""
3. First field (Authors) is UNQUOTED raw text (ends with comma)
4. Fields 2-N are wrapped in ""..."" instead of "..."
5. Trailing semicolons added as padding

THE FIX:
1. Strip trailing semicolons
2. Strip outer wrapper quotes (first and last char)
3. Replace all "" with " (undo quote doubling)
4. Result is valid standard CSV with comma delimiter
'''

import streamlit as st
import pandas as pd
import csv
import io
from pathlib import Path
from typing import List, Dict, Literal


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: LEARN FROM CORRECT FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def learn_from_correct_format(content: str) -> Dict:
    """Parse correct format and extract structural information."""
    df = pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False)
    return {
        'columns': list(df.columns),
        'num_columns': len(df.columns),
        'sample_row': df.iloc[0].to_dict() if len(df) > 0 else {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: DETECT AND FIX INCORRECT FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def detect_format_type(content: str) -> Literal['correct', 'incorrect', 'unknown']:
    """Detect if file is correct or incorrect Scopus CSV format."""
    first_line = content.split('\n')[0] if content else ''

    # Strong incorrect indicators
    if first_line.endswith(';' * 10):
        return 'incorrect'
    if first_line.count('""') > 10 and first_line.startswith('"'):
        return 'incorrect'

    # Correct format indicators
    if first_line.count('","') >= 10 and not first_line.endswith(';'):
        return 'correct'

    return 'unknown'


def fix_incorrect_scopus_csv(content: str) -> str:
    '''
    Fix corrupted Scopus CSV by reversing the corruption pattern.
    
    Corruption: "Authors,""field2"",""field3"",...,""fieldN""";;;;;;;;;;
    Fix:       Authors,"field2","field3",...,"fieldN"
    '''
    lines = content.splitlines()
    if not lines:
        return ""

    fixed_lines = []

    for line in lines:
        line = line.rstrip('\n\r')
        if not line.strip():
            fixed_lines.append("")
            continue

        # Step 1: Remove trailing semicolons
        line = line.rstrip(';')
        if not line:
            fixed_lines.append("")
            continue

        # Step 2: Strip outer wrapper quotes if present
        # The line starts with " and ends with " (before semicolons were stripped)
        if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
            line = line[1:-1]

        # Step 3: Undo quote doubling — replace all "" with "
        # This converts ""field"" to "field", giving valid CSV
        line = line.replace('""', '"')

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


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
    st.markdown('''
    **Two-phase workflow:**
    1. **Learn**: Upload a correctly-formatted Scopus CSV (Category B)
    2. **Convert**: Upload incorrectly-formatted files (Category A/C) to convert them

    **Fix method:**
    - Strip outer quote wrapper and trailing semicolons
    - Replace doubled quotes "" with standard quotes "
    - Result is valid standard CSV with comma delimiter
    ''')

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
                        'num_columns': structure['num_columns']
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
                fixed_content = fix_incorrect_scopus_csv(content)

                # Debug: show first 1000 chars of fixed content
                with st.expander("Debug: Fixed content preview"):
                    st.code(fixed_content[:1000], language="text")

                # Parse as normal CSV
                df = pd.read_csv(io.StringIO(fixed_content), dtype=str, keep_default_na=False)

                # Verify columns match learned structure
                expected_cols = st.session_state.learned_structure['columns']
                missing_cols = [c for c in expected_cols if c not in df.columns]
                extra_cols = [c for c in df.columns if c not in expected_cols]

                if missing_cols:
                    st.warning(f"⚠️ Missing columns: {missing_cols}")
                if extra_cols:
                    st.warning(f"⚠️ Extra columns: {extra_cols}")

                # Reorder columns to match learned structure
                df = df[[c for c in expected_cols if c in df.columns]]

                # Add any missing columns as empty
                for col in missing_cols:
                    df[col] = ''

                # Final reorder to exactly match learned structure
                df = df[expected_cols]

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
