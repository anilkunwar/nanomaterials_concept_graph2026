import streamlit as st
import pandas as pd
import re
import io
import zipfile
from datetime import datetime

def is_malformed_scopus(text: str) -> bool:
    """
    Detects the specific Scopus export glitch.
    Signatures: Broken delimiter ,"" or excessive trailing semicolons ;;;;;
    """
    return ',\"\"' in text or re.search(r';{5,}$', text, re.MULTILINE) is not None

def repair_scopus_csv(text: str) -> str:
    """
    Repairs the malformed Scopus CSV structure by fixing delimiters 
    and stripping trailing garbage.
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    repaired_lines = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # 1. Fix the broken delimiter: ,"" -> ","
        fixed_line = line.replace(',""', '","')
        
        # 2. Remove trailing garbage (semicolons and extra quotes) at the end of the line
        fixed_line = re.sub(r'[";]+$', '', fixed_line)
        
        # 3. Ensure the line has balanced quotes (add closing quote if odd)
        # This fixes the last field which often loses its closing quote due to the trailing garbage
        if fixed_line.count('"') % 2 != 0:
            fixed_line += '"'
            
        repaired_lines.append(fixed_line)
        
    return '\n'.join(repaired_lines)

def process_file(uploaded_file):
    """Reads, repairs if needed, and parses the CSV."""
    try:
        raw_bytes = uploaded_file.getvalue()
        
        # Try utf-8-sig (handles BOM), fallback to latin-1
        try:
            text = raw_bytes.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw_bytes.decode('latin-1')
            
        was_repaired = False
        if is_malformed_scopus(text):
            text = repair_scopus_csv(text)
            was_repaired = True
            
        # Parse to DataFrame to validate structure
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        
        # Generate structured filename (e.g., categoryA_structured.csv)
        base_name = uploaded_file.name.rsplit('.', 1)[0]
        structured_name = f"{base_name}_structured.csv"
        
        return {
            "original_name": uploaded_file.name,
            "structured_name": structured_name,
            "repaired": was_repaired,
            "df": df,
            "csv_text": text,
            "status": "Success"
        }
    except Exception as e:
        return {
            "original_name": uploaded_file.name,
            "structured_name": "N/A",
            "repaired": False,
            "df": None,
            "csv_text": None,
            "status": f"Error: {str(e)}"
        }

def main():
    st.set_page_config(page_title="Scopus CSV Repair & Converter", layout="wide")
    
    st.title("🔧 Scopus CSV Structure Repair Tool")
    st.markdown("""
    This tool automatically detects and repairs malformed Scopus CSV exports (like Category A & C) 
    and converts them into the proper standard structure (like Category B).
    """)
    
    uploaded_files = st.file_uploader("Upload Scopus CSV files", type=['csv'], accept_multiple_files=True)
    
    if uploaded_files:
        results = []
        with st.spinner("Processing and repairing files..."):
            for f in uploaded_files:
                results.append(process_file(f))
                
        st.subheader("📊 Processing Summary")
        
        # Create summary dataframe
        summary_data = []
        for res in results:
            summary_data.append({
                "Original File": res["original_name"],
                "Status": "✅ Repaired" if res["repaired"] else ("✅ Already Clean" if res["status"] == "Success" else f"❌ {res['status']}"),
                "Rows": len(res["df"]) if res["df"] is not None else 0,
                "Columns": len(res["df"].columns) if res["df"] is not None else 0,
                "Structured File": res["structured_name"]
            })
            
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
        
        st.subheader("⬇️ Download Structured Files")
        
        # Individual downloads
        cols = st.columns(min(3, len(results)))
        for i, res in enumerate(results):
            if res["status"] == "Success":
                with cols[i % len(cols)]:
                    st.download_button(
                        label=f"📥 {res['structured_name']}",
                        data=res["csv_text"].encode('utf-8'),
                        file_name=res["structured_name"],
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        # Zip download for all
        if len(results) > 1:
            st.markdown("---")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for res in results:
                    if res["status"] == "Success":
                        zf.writestr(res["structured_name"], res["csv_text"].encode('utf-8'))
                        
            st.download_button(
                label="📦 Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"scopus_structured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
