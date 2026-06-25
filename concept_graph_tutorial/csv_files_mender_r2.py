#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Scopus CSV Format Learner & Converter — Bulletproof Edition v5
===============================================================

THE CORRUPTION PATTERN (observed in Scopus Category A CSV exports):

Correct (Category B):  "Authors","Author full names","Title",...,"EID"
Incorrect (Category A): "Authors,""Author full names"",""Title"",...,""EID""";;;;;;;;;;

The corruption is:
1. Entire row wrapped in extra outer quotes: "...content..."
2. All internal quotes doubled: " -> ""
3. First field (Authors) is UNQUOTED raw text
4. Fields 2-N are wrapped in ""..."" instead of "..."
5. Trailing semicolons added as padding

THE FIX:
Instead of fragile string replacement, we use a state-machine parser that:
- Strips outer wrapper and trailing semicolons
- Extracts the unquoted first field character-by-character
- Parses remaining fields as quoted CSV with "" as quotechar
- Reconstructs a valid standard CSV using Python's csv module

This preserves Abstracts, Titles, and all fields with commas/quotes intact.
'''

import streamlit as st
import pandas as pd
import csv
import io
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal


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
    }

    return structure


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
    Bulletproof fix for corrupted Scopus CSV format.

    Uses state-machine parsing to handle the specific corruption where:
    - Row is wrapped in extra outer quotes
    - Internal quotes are doubled
    - First field is unquoted
    - Remaining fields are wrapped in ""
    '''
    lines = content.splitlines()
    if not lines:
        return ""

    fixed_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            fixed_lines.append("")
            continue

        # Step 1: Remove trailing semicolons
        line = line.rstrip(';')
        if not line:
            fixed_lines.append("")
            continue

        # Step 2: Parse with state machine
        fields = _parse_corrupted_line(line)

        # Step 3: Write as properly formatted CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator='')
        writer.writerow(fields)
        fixed_lines.append(output.getvalue())

    return '\n'.join(fixed_lines)


def _parse_corrupted_line(line: str) -> List[str]:
    '''
    State-machine parser for a single corrupted Scopus CSV line.

    CORRUPTED FORMAT:
    "field1,""field2"",""field3"",...,""fieldN"""

    Where:
    - " at position 0 = outer wrapper start
    - field1 = unquoted text until first ,"" delimiter
    - ,"" = delimiter between fields
    - field2..N = text wrapped in ""...""
    - """ at end = "" (field close) + " (outer wrapper close)
    '''
    fields = []
    current_field = []
    state = 'FIRST_FIELD'  # FIRST_FIELD, IN_QUOTED_FIELD

    i = 0
    while i < len(line):
        char = line[i]
        next_char = line[i + 1] if i + 1 < len(line) else None
        next_next_char = line[i + 2] if i + 2 < len(line) else None

        if state == 'FIRST_FIELD':
            # First field is unquoted raw text until we hit ,"" delimiter
            if char == ',' and next_char == '"' and next_next_char == '"':
                # Found delimiter ,"" - end of first field
                fields.append(''.join(current_field))
                current_field = []
                state = 'IN_QUOTED_FIELD'
                i += 3  # Skip past ,""
            elif i == 0 and char == '"':
                # Opening outer wrapper quote - skip it
                i += 1
            else:
                current_field.append(char)
                i += 1

        elif state == 'IN_QUOTED_FIELD':
            # Inside a quoted field (wrapped in "")
            if char == '"' and next_char == '"':
                # Two consecutive quotes - check what comes after
                # Look ahead to determine if this is:
                # a) Escaped quote inside field: "" followed by non-quote
                # b) End of field: """ followed by ,"" or end
                # c) End of line: """ at very end

                if i + 2 < len(line):
                    char_after = line[i + 2]
                    if char_after == '"':
                        # """ pattern - this is end of field + start of next or end
                        # Check if followed by ,"" (next field) or just end
                        if i + 3 < len(line) and line[i + 3] == ',':
                            # ""","" pattern - end of field, delimiter starts
                            fields.append(''.join(current_field))
                            current_field = []
                            # Skip past """ , then the ,"" will be handled
                            i += 3  # Now at ','
                        elif i + 3 >= len(line):
                            # """ at end of line - end of last field + outer wrapper close
                            fields.append(''.join(current_field))
                            current_field = []
                            i += 3
                        else:
                            # """ followed by something else - treat as escaped quote
                            current_field.append('"')
                            i += 2
                    else:
                        # "" followed by non-quote - this is escaped quote inside field
                        current_field.append('"')
                        i += 2
                else:
                    # "" at end of line - end of field
                    fields.append(''.join(current_field))
                    current_field = []
                    i += 2
            else:
                current_field.append(char)
                i += 1

    # Handle any remaining content
    if current_field:
        fields.append(''.join(current_field))

    return fields


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

    **Robust state-machine parser handles:**
    - Outer quote wrapper on entire rows
    - Doubled internal quotes ("" -> ")
    - Corrupted delimiter (,"" instead of ",")
    - Trailing semicolon spam
    - Unquoted first column (Authors) with commas
    - Abstracts with commas and quotes preserved intact
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

        st.info("🔍 Detected: **Incorrect format** — converting with state-machine parser...")

        try:
            with st.spinner("Converting..."):
                # FIX THE FORMAT using state machine parser
                fixed_content = fix_incorrect_scopus_csv(content)

                # Debug: show first 500 chars of fixed content
                with st.expander("Debug: Fixed content preview"):
                    st.code(fixed_content[:1000], language="text")

                # NOW PARSE AS NORMAL CSV
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
