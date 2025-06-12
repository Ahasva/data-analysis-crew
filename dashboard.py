# ======= dashboard.py ======================================================
"""
Streamlit dashboard for the Data-Analysis-Crew outputs.
Run with   streamlit run dashboard.py
(or let main.py open it automatically after the crew finishes).
"""
import sys
from pathlib import Path
import json
import streamlit as st
import streamlit.web.cli as stcli

# ======= paths ==============================================================
OUTPUT_DIR   = Path("output")
REPORT_MD    = OUTPUT_DIR / "final-insight-summary.md"
REPORT_JSON  = OUTPUT_DIR / "model-report.json"
PLOTS_DIR    = OUTPUT_DIR / "plots"

# ======= page config ========================================================
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("📊 Data-Analysis Crew Report")

# ======= Executive summary ==================================================
st.header("📋 Executive Summary")
if REPORT_MD.exists():
    st.markdown(REPORT_MD.read_text(), unsafe_allow_html=True)
else:
    st.warning("Executive summary not found (expected at `output/final-insight-summary.md`).")

# ======= Model report =======================================================
st.header("🤖 Model Report")

model_data = {}
if REPORT_JSON.exists():
    if REPORT_JSON.stat().st_size > 0:
        try:
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

            # ======= 🔍 Model comparison chart ================================
            all_scores = model_data.get("all_model_scores")
            if isinstance(all_scores, dict) and all_scores:
                st.subheader("📊 Model Comparison")
                st.bar_chart(all_scores)

        except json.JSONDecodeError as e:
            st.error(f"❌ model-report.json is malformed:\n\n{e}")
    else:
        st.warning("⚠️ model-report.json is empty.")
else:
    st.warning("❌ model-report.json not found in `output/`.")

# ======= Visualisations ======================================================
st.header("🖼️ Visualisations")

feature_plot = model_data.get("feature_importance_path")
conf_matrix  = model_data.get("confusion_matrix_path")
problem_type = model_data.get("problem_type")
secondary_plots = model_data.get("secondary_plot_paths", [])

# Feature importances
if feature_plot:
    plot_path = Path(feature_plot)
    if plot_path.exists():
        st.subheader("🔍 Feature Importances")
        st.image(str(plot_path), caption="Feature Importances", use_container_width=True)
    else:
        st.warning(f"Expected plot at {plot_path}, but it was not found.")

# Confusion matrix
if conf_matrix:
    conf_matrix_path = Path(conf_matrix)
    if conf_matrix_path.exists():
        st.subheader("📉 Confusion Matrix")
        st.image(str(conf_matrix_path), caption="Confusion Matrix", use_container_width=True)
    else:
        st.warning(f"Expected plot at {conf_matrix_path}, but it was not found.")

# Secondary plots (ROC, PR, Residuals, etc.)
if secondary_plots:
    for path_str in secondary_plots:
        spath = Path(path_str)
        if spath.exists():
            lower_name = spath.name.lower()
            title = "📊 Secondary Plot"
            caption = spath.name

            if problem_type == "classification":
                if "roc" in lower_name:
                    title = "📈 ROC Curve"
                    caption = "Receiver Operating Characteristic Curve"
                elif "precision" in lower_name:
                    title = "📈 Precision-Recall Curve"
                    caption = "Precision vs Recall"
                elif "confusion" in lower_name:
                    continue  # already displayed

            elif problem_type == "regression":
                if "residual" in lower_name:
                    title = "📉 Residuals Plot"
                    caption = "Residuals vs. Prediction"

            st.subheader(title)
            st.image(str(spath), caption=caption, use_container_width=True)
        else:
            st.warning(f"Missing secondary plot: {spath}")

# ======= Downloads ===========================================================
st.header("📂 Download Outputs")
col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.subheader("Executive Summary Debug")
    st.code(REPORT_MD.read_text() if REPORT_MD.exists() else "❌ File not found!")

    if REPORT_MD.exists():
        st.download_button("📥 Executive Summary (.md)",
                           REPORT_MD.read_bytes(),
                           file_name="summary.md")
    else:
        st.error("File not found: output/final-insight-summary.md")

with col_dl2:
    if REPORT_JSON.exists():
        st.download_button("📥 Model Report (.json)",
                           REPORT_JSON.read_bytes(),
                           file_name="model-report.json")

# ======= allow `python dashboard.py` ========================================
if __name__ == "__main__" and not st.runtime.exists():
    sys.argv = ["streamlit", "run", __file__]
    stcli.main()
