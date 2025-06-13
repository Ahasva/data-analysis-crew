#!/usr/bin/env python
"""
Entry-point for the Data-Analysis Crew pipeline
Run with  `crewai run`  or  `python -m data_analysis_crew.main`
"""
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone
from data_analysis_crew.utils.instructions import INSTALL_LIB_TEMPLATE, AVAILABLE_LIBRARIES
from data_analysis_crew.crew import DataAnalysisCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ─────────────────────────────────────── variables ───────────────────────────────────────
FILE_NAME = "diabetes.csv"
ROOT_FOLDER = "knowledge"

# Exposition of variables for import by other modules (e.g. crew.py)
__all__ = ["FILE_NAME", "ROOT_FOLDER", "REL_PATH_DATA", "run"]

# ───────────────────────────────────────── paths ─────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DATA_FILE     = PROJECT_ROOT / ROOT_FOLDER / FILE_NAME
REL_PATH_DATA = DATA_FILE.relative_to(PROJECT_ROOT)

OUTPUT_DIR_ABS = PROJECT_ROOT / "output"
PLOT_PATH_ABS  = OUTPUT_DIR_ABS / "plots"

# These are the relative paths passed into the crew
OUTPUT_DIR_REL = OUTPUT_DIR_ABS.relative_to(PROJECT_ROOT)
PLOT_PATH_REL  = PLOT_PATH_ABS.relative_to(PROJECT_ROOT)

# Create necessary folders
OUTPUT_DIR_ABS.mkdir(parents=True, exist_ok=True)
PLOT_PATH_ABS.mkdir(parents=True, exist_ok=True)

# Ensure dashboard.py exists
DASHBOARD_PY = PROJECT_ROOT / "dashboard.py"
if not DASHBOARD_PY.exists():
    raise FileNotFoundError(f"dashboard.py expected at {DASHBOARD_PY}")

# ───────────────────────────────────────── prompt ────────────────────────────────────────
REQUEST = """
What are the main factors for Diabetes?
Which feature in the given data has the gravest impact on the patient,
resulting in diabetes?
"""

# Core model types to suggest — not dictate
AVAILABLE_MODELS = {
    "classification": [
        "logistic_reg",           # Logistic Regression
        "random_forest",          # Random Forest
        "svm",                    # Support Vector Machine
        "knn",                    # K-Nearest Neighbors
        "gradient_boosted_trees", # Gradient Boosting (e.g. XGBoost, LightGBM)
        "naive_bayes",            # Gaussian / Multinomial Naive Bayes
        "decision_tree",          # Decision Tree
        "mlp_classifier",         # Neural network classifier
    ],
    "regression": [
        "linear_regression",      # Ordinary Least Squares
        "ridge",                  # Ridge Regression (L2)
        "lasso",                  # Lasso Regression (L1)
        "elastic_net",            # Combined L1 + L2
        "random_forest",          # Ensemble method
        "svm",                    # Support Vector Regression
        "gradient_boosted_trees", # Boosted Regression Trees
        "mlp_regressor",          # Neural Network regressor
        "k_neighbors_regressor",  # KNN for regression
    ]
}

# Metrics by problem type
METRICS_BY_TYPE = {
    "classification": [
      "accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"
    ],
    "regression": [
      "r2_score", "mean_squared_error", "mean_absolute_error", "rmse"
    ]
}

# Required plots for visual validation (dashboard)
EXPECTED_PLOTS = [
    "feature_importances.png",
    "confusion_matrix.png",
    "roc_curve.png",
    "precision_recall.png",
    "model_score_comparison.png"
]


# ─────────────────────────────────────── main helpers ─────────────────────────────────────
def run() -> None:
    """Run the full data-analysis crew pipeline."""
    inputs = {
        "file_name"             : FILE_NAME,
        "root_folder"           : ROOT_FOLDER,
        "dataset_path"          : str(REL_PATH_DATA),
        "raw_path"              : str(REL_PATH_DATA),
        "cleaned_path"          : f"{ROOT_FOLDER}/{FILE_NAME[:-4]}_cleaned.csv",
        "request"               : REQUEST,
        "output_dir"            : str(OUTPUT_DIR_REL),
        "plot_path"             : str(PLOT_PATH_REL),
        "install_hint"          : INSTALL_LIB_TEMPLATE,
        "available_libraries"   : AVAILABLE_LIBRARIES,
        "datetime"              : datetime.now(timezone.utc).isoformat(),
        "classification_models" : AVAILABLE_MODELS["classification"],
        "regression_models"     : AVAILABLE_MODELS["regression"],
        "classification_metrics": METRICS_BY_TYPE["classification"],
        "regression_metrics"    : METRICS_BY_TYPE["regression"],
        "expected_plots"        : EXPECTED_PLOTS
    }

    try:
        crew = DataAnalysisCrew().crew()

        print("\n🧭 Planned task order:")
        for t in crew.tasks:
            print(f"→ {t.agent.role:<22} : {t.description.splitlines()[0]}")

        print("🧾 Passing dataset:", inputs["dataset_path"])
        print("📁 Output path    :", inputs["output_dir"])
        print("📂 Plot path      :", inputs["plot_path"])

        result = crew.kickoff(inputs=inputs)
        print("🚀 Dashboard launch has been delegated to the crew.")

        return result

    except Exception as e:
        raise RuntimeError(f"[RUN ERROR] Crew execution failed: {e}") from e

def train():
    """
    Train the crew with synthetic prompt and iterations.
    """
    inputs = {
        "data_source": "https://example.com/data.csv",
        "dataset_path": "https://example.com/data.csv",
        "request": "What are the main sales trends over the last 6 months?",
        "output_dir": "output",
    }

    try:
        DataAnalysisCrew().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise RuntimeError(f"[TRAIN ERROR] Failed to train the crew: {e}") from e

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        DataAnalysisCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise RuntimeError(f"[REPLAY ERROR] Failed to replay the task: {e}") from e

def test():
    """
    Test the crew execution and return results.
    """
    inputs = {
        "data_source": "https://example.com/data.csv",
        "dataset_path": "https://example.com/data.csv",
        "request": "What are the main sales trends over the last 6 months?",
        "output_dir": "output"
    }

    try:
        DataAnalysisCrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise RuntimeError(f"[TEST ERROR] Failed to test the crew: {e}") from e

def validate_final_summary(output_path: Path):
    """Simple checklist validator for final summary report."""
    summary_file = output_path / "final-insight-summary.md"
    if not summary_file.exists():
        print("❌ final-insight-summary.md not found.")
        return

    content = summary_file.read_text(encoding="utf-8").lower()

    checks = {
        "✔ summary section"      : "executive summary" in content,
        "✔ bullet insights"      : "-" in content or "*" in content,
        "✔ embedded visual"      : "<img " in content,
        "✔ metrics present"      : any(metric in content for metric in METRICS_BY_TYPE["classification"] + METRICS_BY_TYPE["regression"]),
        "✔ dashboard mentioned"  : "dashboard" in content or "http://localhost" in content
    }

    print("\n🔍 Final Report Checklist:")
    log_path = output_path / "final-checklist.log"
    with log_path.open("w", encoding="utf-8") as f:
        for desc, passed in checks.items():
            result = '✅' if passed else '❌'
            print(f"{desc:<30} {result}")
            f.write(f"{desc:<28} {'OK' if passed else 'MISSING'}\n")
    print(f"📝 Checklist log saved to: {log_path}")

# ─────────────────────────────── script launch ───────────────────────────────
if __name__ == "__main__":
    run()
