# dashboard.py  ──────────────────────────────────────────────────────────────
"""
Streamlit dashboard for the Data-Analysis-Crew outputs.
Run with   streamlit run dashboard.py
(or let main.py open it automatically after the crew finishes).
"""
from pathlib import Path
import json
import streamlit as st

# ───────────────────────── paths ─────────────────────────
OUTPUT_DIR   = Path("output")
REPORT_MD    = OUTPUT_DIR / "final-insight-summary.md"
REPORT_JSON  = OUTPUT_DIR / "model-report.json"
PLOTS_DIR    = OUTPUT_DIR / "plots"           # kept for completeness

# ──────────────────────── page config ────────────────────
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("📊 Data-Analysis Crew Report")

# ═════════════════════ Executive summary ═════════════════════
st.header("📋 Executive Summary")
if REPORT_MD.exists():
    st.markdown(REPORT_MD.read_text(), unsafe_allow_html=True)
else:
    st.warning("Executive summary not found (expected at `output/final-insight-summary.md`).")

# ═════════════════════ Model report ═════════════════════
st.header("🤖 Model Report")

model_data = {}
if REPORT_JSON.exists():
    model_data = json.loads(REPORT_JSON.read_text())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Info")
        st.markdown(f"**Model Type:** {model_data.get('model_type', 'N/A')}")
        st.markdown(f"**Target Variable:** `{model_data.get('target', 'N/A')}`")

    with col2:
        st.subheader("Plain Summary")
        st.markdown(model_data.get("plain_summary", "No summary found."))

    st.subheader("📈 Metrics")
    for metric, value in (model_data.get("metrics") or {}).items():
        st.markdown(f"- **{metric}**: `{value:.4f}`")
else:
    st.warning("`model-report.json` not found in `output/`.")

# ═════════════════════ Visualisations ═════════════════════
st.header("🖼️ Visualisations")

feature_plot = model_data.get("feature_importance_path")
conf_matrix  = model_data.get("confusion_matrix_path")

if feature_plot and Path(feature_plot).exists():
    st.subheader("🔍 Feature Importances")
    st.image(feature_plot, caption="Feature Importances", use_column_width=True)
else:
    st.info("Feature-importance plot not found.")

if conf_matrix and Path(conf_matrix).exists():
    st.subheader("📉 Confusion Matrix")
    st.image(conf_matrix, caption="Confusion Matrix", use_column_width=True)

# ═════════════════════ Downloads ═════════════════════
st.header("📂 Download Outputs")
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    if REPORT_MD.exists():
        st.download_button("📥 Executive Summary (.md)",
                           REPORT_MD.read_bytes(),
                           file_name="summary.md")
with col_dl2:
    if REPORT_JSON.exists():
        st.download_button("📥 Model Report (.json)",
                           REPORT_JSON.read_bytes(),
                           file_name="model-report.json")

# ─────────────────── allow `python dashboard.py` ───────────────────
if __name__ == "__main__":
    import sys, streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", __file__]
    stcli.main()
