# ======= dashboard.py ======================================================
"""
Streamlit dashboard for the Data-Analysis-Crew outputs.
Run with   streamlit run dashboard.py
(or let main.py open it automatically after the crew finishes).
"""
import sys
import re
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.web.cli as stcli

# ======= paths ==============================================================
OUTPUT_DIR         = Path("output")
REPORT_MD          = OUTPUT_DIR / "final-insight-summary.md"
REPORT_JSON        = OUTPUT_DIR / "model-report.json"
TECHNICAL_METRICS  = OUTPUT_DIR / "technical-metrics.md"

# ======= page config ========================================================
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("📊 Data-Analysis Crew Report")

# ======= Preload technical-metrics.md content once ==========================
tm_text = TECHNICAL_METRICS.read_text() if TECHNICAL_METRICS.exists() else ""

# ======= Executive summary ==================================================
st.header("📋 Executive Summary")

if REPORT_MD.exists():
    try:
        report_data = json.loads(REPORT_MD.read_text())
        raw_markdown = report_data.get("summary", "")
    except json.JSONDecodeError:
        # fallback if it's a real markdown file
        raw_markdown = REPORT_MD.read_text()

    # 1) Extract the Embedded Visuals and Metrics Summary blocks
    emb_match = re.search(
        r'##\s*✅\s*Embedded Visuals\s*\n([\s\S]*?)(?=\n##\s*✅|\Z)',
        raw_markdown
    )
    emb_block = emb_match.group(1) if emb_match else None

    metrics_match = re.search(
        r'##\s*✅\s*Metrics Summary\s*\n([\s\S]*?)(?=\n##\s*✅|\Z)',
        raw_markdown
    )
    metrics_block = metrics_match.group(1) if metrics_match else None

    # 2) Remove unwanted sections and artifacts from raw markdown
    cleaned = raw_markdown
    cleaned = re.sub(r'(?m)^##\s*✅\s*Embedded Visuals[\s\S]*?(?=\n##|\Z)', "", cleaned)
    cleaned = re.sub(r'(?m)^##\s*✅\s*Metrics Summary[\s\S]*?(?=\n##|\Z)', "", cleaned)
    cleaned = re.sub(r'(?m)^##\s*✅\s*Final Checklist[\s\S]*', "", cleaned)
    cleaned = re.sub(r'(?m)^##\s*✅.*$', "", cleaned)
    cleaned = re.sub(r'<img\s+src="[^"]+"[^>]*>', "", cleaned)
    cleaned = re.sub(r'!\[.*?\]\([^)]+\)', "", cleaned)
    cleaned = re.sub(r'(?m)^#+\s*Plots.*$', "", cleaned)
    cleaned = re.sub(r'(?m)^#+\s*(Feature Importance|Confusion Matrix|ROC Curve).*$','', cleaned)
    cleaned = re.sub(r'(?m)^\s*[-*]\s.*[✔✖].*$', "", cleaned)
    cleaned = re.sub(r'\n{3,}', "\n\n", cleaned).strip()
    cleaned = re.sub(r'(?m)^#\s*Plots[\s\S]*?(?=^#|\Z)', "", cleaned)

    # 3) Split into narrative, key insights, recommendation
    lines = cleaned.splitlines()
    bullet_start = next((i for i, L in enumerate(lines) if L.lstrip().startswith("- ")), len(lines))
    narrative = "\n".join(lines[:bullet_start]).strip()
    end = bullet_start
    while end < len(lines) and (lines[end].lstrip().startswith("- ") or not lines[end].strip()):
        end += 1
    bullets = lines[bullet_start:end]
    recommendation = "\n".join(lines[end:]).strip()

    # Render narrative
    if narrative:
        st.markdown(narrative, unsafe_allow_html=True)

    # Render Key Insights
    if bullets:
        st.subheader("✅ Key Insights")
        st.markdown("\n".join(bullets), unsafe_allow_html=True)

    # Render Recommendation
    if recommendation:
        st.subheader("✅ Recommendation")
        st.markdown(recommendation, unsafe_allow_html=True)

    # Render Embedded Visuals
    if emb_block:
        st.subheader("✅ Embedded Visuals")
        for line in emb_block.splitlines():
            line = line.strip()
            img_match = re.search(r'<img\s+src="([^"]+)"', line)
            if img_match:
                rel = img_match.group(1).strip().lstrip("./")
                img_path = (OUTPUT_DIR / rel).resolve()
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"} and img_path.exists():
                    st.image(str(img_path), width=480)
                else:
                    st.warning(f"⚠️ Image not found or unsupported: {img_path}")
            elif line.startswith("-"):
                st.markdown(line)

    # Render Metrics Summary
    if metrics_block:
        st.subheader("📊 Metrics Summary")
        for line in metrics_block.splitlines():
            line = line.strip()
            if line.startswith("-"):
                st.markdown(line)

else:
    st.warning("❌ Executive summary not found (`output/final-insight-summary.md`).")

# ======= Model report ======================================================
st.header("🤖 Model Report")

# Load model-report.json
model_data = {}
if REPORT_JSON.exists():
    txt = REPORT_JSON.read_text().strip()
    if txt:
        try:
            model_data = json.loads(txt)
        except json.JSONDecodeError as e:
            st.error(f"❌ Could not parse model-report.json:\n\n{e}")
    else:
        st.warning("⚠️ model-report.json is empty.")
else:
    st.warning("❌ model-report.json not found.")

# Display core info
if model_data:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Model Info (best performing model)")
        st.markdown(f"**Model Type:** {model_data.get('model_type','N/A')}")
        st.markdown(f"**Target Variable:** `{model_data.get('target','N/A')}`")
    with col2:
        st.subheader("Plain Summary")
        st.markdown(model_data.get("plain_summary","No summary found."))

    # Metrics
    if model_data.get("metrics"):
        st.subheader("📈 Metrics")
        for nm, val in model_data["metrics"].items():
            st.markdown(f"- **{nm.title()}**: `{val:.4f}`")

    # Model comparison chart
    if isinstance(model_data.get("all_model_scores"), dict):
        st.subheader("📊 Model Comparison")
        st.bar_chart(model_data["all_model_scores"])

    # Hyperparameters
    if tm_text:
        hp = re.search(r'```json\s*(\{[\s\S]+?\})\s*```', tm_text)
        if hp:
            try:
                best_params = json.loads(hp.group(1))
                model_data["best_params"] = best_params
            except json.JSONDecodeError:
                st.warning("⚠️ Could not parse JSON hyperparameters.")
    if model_data.get("best_params"):
        st.subheader("⚙️ Hyperparameters")
        st.json(model_data["best_params"])

    # Other Model Scores
    if tm_text:
        lines = tm_text.splitlines()
        start = next((i for i, L in enumerate(lines) if L.strip().startswith("## All Model Scores")), None)
        if start is not None:
            rows = []
            for L in lines[start+1:]:
                if not L.strip():
                    continue
                if L.strip().startswith("|"):
                    rows.append(L)
                else:
                    break
            scores = {}
            for r in rows[2:]:
                cols = [c.strip() for c in r.split("|")[1:-1]]
                if len(cols) == 2:
                    m, scr = cols
                    try:
                        scores[m] = float(scr)
                    except ValueError:
                        pass
            if scores:
                st.subheader("🏆 Other Model Scores")
                df_scores = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])
                df_scores.index.name = "Model"
                df_scores["Score"] = df_scores["Score"].map("{:.4f}".format)
                st.table(df_scores)

# ======= Visualisations ======================================================
st.header("🖼️ Visualisations")

plot_dir = OUTPUT_DIR / "plots"
all_images = sorted([
    img for img in plot_dir.glob("*.png")
    if img.is_file()
]) if plot_dir.exists() else []

# Keywords to detect important plots and their display names
important_keywords = {
    "roc": "📈 ROC Curve",
    "correlation": "🌐 Correlation Heatmap",
    "feature_importance": "🔍 Feature Importances",
    "confusion_matrix": "🧮 Confusion Matrix",
    "precision": "📈 Precision-Recall Curve",
    "residual": "📉 Residuals Plot",
    "comparison": "📊 Model Comparison"
}

important_imgs = []
dist_imgs = []

# Categorize images
for img in all_images:
    fname = img.name.lower()

    if fname.startswith("distribution_"):
        title = fname.replace("distribution_", "").replace(".png", "").replace("_", " ").title()
        dist_imgs.append((img, title))
    else:
        matched_title = None
        for kw, label in important_keywords.items():
            if kw in fname:
                matched_title = label
                break
        title = matched_title or "📊 Other Plot"
        important_imgs.append((img, title))

# Show important plots (2 per row, full width)
if important_imgs:
    st.subheader("📌 Key Visualisations")
    for i in range(0, len(important_imgs), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(important_imgs):
                with cols[j]:
                    st.subheader(important_imgs[idx][1])
                    st.image(str(important_imgs[idx][0]), use_container_width=True)

# Show distribution plots (3 per row, small size)
if dist_imgs:
    st.subheader("🔬 Distribution Plots")
    for i in range(0, len(dist_imgs), 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            if idx < len(dist_imgs):
                with cols[j]:
                    st.caption(dist_imgs[idx][1])
                    st.image(str(dist_imgs[idx][0]), width=320)

# ======= Download Outputs ==================================================
st.header("📂 Download Outputs")
dl1, dl2 = st.columns(2)
with dl1:
    st.subheader("Executive Summary Debug")
    if REPORT_MD.exists():
        # Try to extract raw Markdown (from JSON or plain md)
        try:
            parsed = json.loads(REPORT_MD.read_text())
            summary_md = parsed.get("summary", REPORT_MD.read_text())
        except json.JSONDecodeError:
            summary_md = REPORT_MD.read_text()

        with st.expander("🔍 Show raw Markdown content"):
            st.code(summary_md)

        st.download_button("📥 Executive Summary (.md)",
                           summary_md,
                           file_name="summary.md")
    else:
        st.error("File not found: final-insight-summary.md")

with dl2:
    if REPORT_JSON.exists():
        st.download_button("📥 Model Report (.json)",
                           REPORT_JSON.read_bytes(),
                           file_name="model-report.json")

# ======= allow `python dashboard.py` ========================================
if __name__ == "__main__" and not st.runtime.exists():
    sys.argv = ["streamlit", "run", __file__]
    stcli.main()
