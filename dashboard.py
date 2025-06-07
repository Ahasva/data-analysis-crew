import os
import json
import streamlit as st
from pathlib import Path

# Project structure
OUTPUT_DIR = Path("output")
REPORT_MD = OUTPUT_DIR / "final-insight-summary.md"
REPORT_JSON = OUTPUT_DIR / "model-report.json"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Page setup
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("📊 Data Analysis Crew Report")

# --- Executive Summary ---
st.header("📋 Executive Summary")
if REPORT_MD.exists():
    with REPORT_MD.open("r") as f:
        summary_md = f.read()
    st.markdown(summary_md, unsafe_allow_html=True)
else:
    st.warning("Executive summary not found.")

# --- Model Report ---
st.header("🤖 Model Report")

if REPORT_JSON.exists():
    with REPORT_JSON.open("r") as f:
        model_data = json.load(f)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Info")
        st.markdown(f"**Model Type:** {model_data.get('model_type', 'N/A')}")
        st.markdown(f"**Target Variable:** `{model_data.get('target', 'N/A')}`")

    with col2:
        st.subheader("Executive Summary")
        st.markdown(model_data.get("plain_summary", "No summary found."))

    st.subheader("📈 Model Metrics")
    for metric, value in model_data.get("metrics", {}).items():
        st.markdown(f"- **{metric}**: `{value:.4f}`")

else:
    st.warning("Model report JSON not found.")

# --- Visualizations ---
st.header("🖼️ Visualizations")

feature_plot = model_data.get("feature_importance_path", None)
conf_matrix = model_data.get("confusion_matrix_path", None)

if feature_plot and Path(feature_plot).exists():
    st.subheader("🔍 Feature Importances")
    st.image(feature_plot, caption="Feature Importances", use_column_width=True)
else:
    st.info("Feature importance plot not found.")

if conf_matrix and Path(conf_matrix).exists():
    st.subheader("📉 Confusion Matrix")
    st.image(conf_matrix, caption="Confusion Matrix", use_column_width=True)

# --- Files ---
st.header("📂 Download Outputs")
st.download_button("📥 Download Executive Summary (.md)", REPORT_MD.read_bytes(), file_name="summary.md")
st.download_button("📥 Download Model Report (.json)", REPORT_JSON.read_bytes(), file_name="model-report.json")