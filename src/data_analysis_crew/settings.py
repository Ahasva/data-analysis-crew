# src/data_analysis_crew/settings.py

# ────────────────────────────────── used data variables ──────────────────────────────────
ROOT_FOLDER = "knowledge"
FILE_NAME = "diabetes.csv"

DASHBOARD_FILE = "dashboard.py"

# 🚨 Final relative input path
REL_PATH_DATA = f"{ROOT_FOLDER}/{FILE_NAME}"

# Clean output paths — used in tasks/main.py, not as actual path objects
OUTPUT_DIR = "output"
PLOT_PATH = "output/plots"

# ───────────────────────────────────────── prompt ────────────────────────────────────────
REQUEST = """
What are the main factors for Diabetes?
Which feature in the given data has the gravest impact on the patient,
resulting in diabetes?
"""

# ───────────────────────────────── models, metrics, plots ────────────────────────────────
AVAILABLE_MODELS = {
    "classification": [
        "logistic_reg", "random_forest", "svm", "knn",
        "gradient_boosted_trees", "naive_bayes", "decision_tree", "mlp_classifier"
    ],
    "regression": [
        "linear_regression", "ridge", "lasso", "elastic_net",
        "random_forest", "svm", "gradient_boosted_trees",
        "mlp_regressor", "k_neighbors_regressor"
    ]
}

METRICS_BY_TYPE = {
    "classification": ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"],
    "regression": ["r2_score", "mean_squared_error", "mean_absolute_error", "rmse"]
}

EXPECTED_PLOTS = [
    "feature_importances.png", "confusion_matrix.png", "roc_curve.png",
    "precision_recall.png", "model_score_comparison.png"
]

# ───────────────────────────────────────── __all__ ────────────────────────────────────────
__all__ = [
    "ROOT_FOLDER", "FILE_NAME", "REL_PATH_DATA", "OUTPUT_DIR", "PLOT_PATH",
    "REQUEST", "AVAILABLE_MODELS", "METRICS_BY_TYPE", "EXPECTED_PLOTS"
]
