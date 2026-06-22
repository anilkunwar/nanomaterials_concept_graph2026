import streamlit as st
import pandas as pd
import re
import io
import csv

def clean_malformed_csv(raw_text: str) -> str:
    """
    Repairs the specific Scopus export bug where the entire row is wrapped in quotes,
    internal quotes are doubled, and there are hundreds of trailing semicolons.
    """
    lines = raw_text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # 1. Strip massive trailing semicolons (5 or more) and whitespace.
        # We use {5,} to ensure we don't accidentally strip a legitimate single semicolon.
        line = re.sub(r';{5,}\s*$', '', line)
        
        # 2. Unwrap outer quotes if the entire line is enclosed in them
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
            
        # 3. Unescape doubled quotes ("" -> ")
        line = line.replace('""', '"')
        
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines)

def main():
    st.set_page_config(page_title="Scopus CSV Schema Learner & Fixer", layout="wide", page_icon="🏗️")
    
    st.title("🏗️ Scopus CSV Schema Learner & Fixer")
    st.markdown("""
    This app works in a two-step process to fix malformed Scopus CSV exports:
    1. **Learn**: Upload a correctly formatted Scopus CSV to define the target schema.
    2. **Fix**: Upload the malformed CSV files. The app will clean them and map them exactly to the learned schema.
    """)
    
    st.divider()
    
    # ==========================================
    # STEP 1: LEARN TARGET SCHEMA
    # ==========================================
    st.header("Step 1: Define Target Schema")
    st.caption("Upload a correctly formatted Scopus CSV file (e.g., Category B) to establish the target column structure.")
    
    ref_file = st.file_uploader("Upload Reference (Correct) CSV", type=['csv'], key="ref_uploader")
    
    if ref_file is not None:
        if st.button("🧠 Learn Format from Reference File", type="primary"):
            try:
                # Use utf-8-sig to handle potential BOM (Byte Order Mark) from Windows exports
                ref_df = pd.read_csv(ref_file, dtype=str, keep_default_na=False, encoding='utf-8-sig')
                st.session_state['target_schema'] = list(ref_df.columns)
                st.success(f"✅ Successfully learned schema with **{len(ref_df.columns)}** columns.")
                with st.expander("View Learned Columns"):
                    st.write(st.session_state['target_schema'])
            except Exception as e:
                st.error(f"Failed to parse reference file: {e}")
                
    # Stop execution if schema hasn't been learned yet
    if 'target_schema' not in st.session_state:
        st.info("👆 Please complete Step 1 by uploading and learning from a correctly formatted CSV file.")
        st.stop()
        
    # ==========================================
    # STEP 2: FIX MALFORMED FILES
    # ==========================================
    st.divider()
    st.header("Step 2: Upload & Fix Malformed Files")
    st.caption("Upload the malformed Scopus CSV files (e.g., Category A/C). They will be cleaned and mapped to the learned schema.")
    
    malformed_files = st.file_uploader(
        "Upload Malformed CSV(s)", 
        type=['csv'], 
        accept_multiple_files=True, 
        key="mal_uploader"
    )
    
    if malformed_files:
        target_schema = st.session_state['target_schema']
        
        for m_file in malformed_files:
            with st.expander(f"📄 Processing: {m_file.name}", expanded=True):
                try:
                    # Read raw bytes and handle encoding
                    raw_bytes = m_file.read()
                    try:
                        raw_text = raw_bytes.decode('utf-8-sig')
                    except UnicodeDecodeError:
                        raw_text = raw_bytes.decode('latin-1')
                        
                    # Clean the text using the repair pipeline
                    cleaned_text = clean_malformed_csv(raw_text)
                    
                    # Parse cleaned text into a DataFrame
                    df = pd.read_csv(io.StringIO(cleaned_text), dtype=str, keep_default_na=False)
                    
                    st.success(f"✅ Cleaned and parsed **{len(df):,}** rows.")
                    
                    # Map to target schema
                    # 1. Add missing columns with empty strings to prevent KeyError
                    for col in target_schema:
                        if col not in df.columns:
                            df[col] = ""
                            
                    # 2. Reorder columns to match target schema exactly
                    # Keep any extra columns that aren't in the target schema at the end
                    extra_cols = [c for c in df.columns if c not in target_schema]
                    final_cols = target_schema + extra_cols
                    df = df[final_cols]
                    
                    st.info(f"📊 Mapped to target schema. Columns: {len(target_schema)} required, {len(extra_cols)} extra.")
                    
                    # Preview the first 3 rows
                    st.dataframe(df.head(3), use_container_width=True, hide_index=True)
                    
                    # Export to CSV
                    csv_buffer = io.StringIO()
                    # Use QUOTE_ALL to ensure strict CSV compliance matching the correct format
                    df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_ALL)
                    csv_bytes = csv_buffer.getvalue().encode('utf-8')
                    
                    # Format filename
                    base_name = m_file.name.replace('.csv', '').replace(' ', '_')
                    download_name = f"{base_name}_structured.csv"
                    
                    st.download_button(
                        label=f"⬇️ Download Fixed CSV ({download_name})",
                        data=csv_bytes,
                        file_name=download_name,
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error processing {m_file.name}: {str(e)}")

if __name__ == "__main__":
    main()
