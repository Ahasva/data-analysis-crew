# src/data_analysis_crew/settings.py

from pathlib import Path

# ────────────────────────────────── used data variables ──────────────────────────────────
ROOT_FOLDER = "knowledge"
FILE_NAME = "diabetes.csv"

# ───────────────────────────────────────── prompt ────────────────────────────────────────
REQUEST = """
What are the main factors for Diabetes?
Which feature in the given data has the gravest impact on the patient,
resulting in diabetes?
"""

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

# ───────────────────────────────── models, metrics, plots ────────────────────────────────
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

# ───────────────────────────────────────── __all__ ────────────────────────────────────────
__all__ = [
    "ROOT_FOLDER",
    "FILE_NAME",
    "REL_PATH_DATA",
    "OUTPUT_DIR_REL",
    "PLOT_PATH_REL",
    "REQUEST",
    "AVAILABLE_MODELS",
    "METRICS_BY_TYPE",
    "EXPECTED_PLOTS",
]