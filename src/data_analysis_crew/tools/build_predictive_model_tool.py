"""
Tool for model builder agent to properly create ML models
"""
# ── src/data_analysis_crew/tools/build_predictive_model_tool.py ──────────────
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from crewai.tools import tool
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, ConfusionMatrixDisplay,
    r2_score, mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.inspection import permutation_importance

# --------------------------------------------------------------------------- #
#  Registry of available models                                               #
# --------------------------------------------------------------------------- #
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "classification": {
        "random_forest": RandomForestClassifier,
        "logistic_reg" : LogisticRegression,
        "svm"          : SVC,
        "knn"          : KNeighborsClassifier,
        "gbt"          : GradientBoostingClassifier,
    },
    "regression": {
        "random_forest": RandomForestRegressor,
        "linear_reg"   : LinearRegression,
        "svm"          : SVR,
        "gbt"          : GradientBoostingRegressor,
    },
}

# --------------------------------------------------------------------------- #
#  Helper: feature importances / coefficients / permutation                   #
# --------------------------------------------------------------------------- #
def _get_importances(model, X_test, y_test, random_state: int = 42):
    """Return (importances, label) or (None, None) if the model offers none."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_, "Built-in"
    if hasattr(model, "coef_"):
        coef = model.coef_.ravel() if model.coef_.ndim == 2 else model.coef_
        return abs(coef), "Coefficient"
    # fallback – permutation (slow)
    try:
        res = permutation_importance(
            model, X_test, y_test, n_repeats=15, random_state=random_state
        )
        return res.importances_mean, "Permutation"
    except Exception:  # pragma: no cover
        return None, None

# --------------------------------------------------------------------------- #
#  Main tool                                                                  #
# --------------------------------------------------------------------------- #
@tool("build_predictive_model")
def build_predictive_model(
    data: pd.DataFrame | str | Path,         # ← accepts DataFrame or path
    *,
    target: str = "outcome",
    problem_type: str | None = None,        # "classification" | "regression"
    model_name: str = "random_forest",
    out_dir: str | Path = "output",
    **model_kwargs,
) -> dict:
    """
    Train a model and save:

      • output/model-report.json  
      • output/final-insight-summary.md  
      • output/plots/feature_importances.png (if available)  
      • output/plots/feature_importance.png **or** residuals.png

    Parameters
    ----------
    data         : DataFrame or str/Path to a CSV containing the target.
    target       : Target column name (default: "outcome").
    problem_type : "classification" | "regression" | None (auto-infer).
    model_name   : See registry above (e.g. "svm", "gbt").
    out_dir      : Root folder for artefacts.
    **model_kwargs : Passed straight to the model constructor.

    Returns
    -------
    dict : Same payload written to *model-report.json*
    """
    # 0) ingest
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("`data` must be a pandas DataFrame or a CSV path/string")

    out_dir   = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_json = out_dir / "model-report.json"
    summary_md = out_dir / "final-insight-summary.md"

    # 1) infer problem type
    if problem_type is None:
        problem_type = (
            "classification"
            if (df[target].dtype == "O" or df[target].nunique() <= 10)
            else "regression"
        )

    if problem_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown problem_type {problem_type!r}")
    if model_name not in _MODEL_REGISTRY[problem_type]:
        valid = ", ".join(_MODEL_REGISTRY[problem_type])
        raise ValueError(f"{model_name!r} invalid for {problem_type}. Valid: {valid}")

    # 2) split data
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    # 3) train model
    ModelCls = _MODEL_REGISTRY[problem_type][model_name]
    default_kwargs = {"random_state": 42} if "random_state" in ModelCls().__dict__ else {}
    model = ModelCls(**{**default_kwargs, **model_kwargs})
    model.fit(X_tr, y_tr)

    # 4) evaluate
    y_pred = model.predict(X_te)
    if problem_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_te, y_pred),
            "f1"      : f1_score(y_te, y_pred, average="weighted"),
        }
    else:
        metrics = {
            "r2"  : r2_score(y_te, y_pred),
            "mse" : mean_squared_error(y_te, y_pred),
        }

    # 5) plots
    feat_png      = plots_dir / "feature_importances.png"
    secondary_png = plots_dir / (
        "confusion_matrix.png" if problem_type == "classification" else "residuals.png"
    )

    importances, imp_label = _get_importances(model, X_te, y_te)
    feat_path_str = None
    if importances is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(X.columns, importances)
        ax.set_xlabel(f"Importance ({imp_label})")
        ax.set_title("Feature Importances")
        fig.tight_layout()
        fig.savefig(feat_png)
        plt.close(fig)
        feat_path_str = str(feat_png)

    if problem_type == "classification":
        disp = ConfusionMatrixDisplay.from_estimator(model, X_te, y_te)
        disp.figure_.savefig(secondary_png)
        plt.close(disp.figure_)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_te, y_pred, alpha=0.6)
        ax.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "k--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Residual Plot")
        fig.tight_layout()
        fig.savefig(secondary_png)
        plt.close(fig)

    # 6) assemble report
    plain = (
        f"Accuracy {metrics['accuracy']:.2%}, F1 {metrics['f1']:.2%}"
        if problem_type == "classification"
        else f"R² {metrics['r2']:.3f}, MSE {metrics['mse']:.4f}"
    )
    report = {
        "model_type"             : ModelCls.__name__,
        "problem_type"           : problem_type,
        "target"                 : target,
        "metrics"                : metrics,
        "plain_summary"          : plain,
        "feature_importance_path": feat_path_str,
        "secondary_plot_path"    : str(secondary_png),
        **({"confusion_matrix_path": str(secondary_png)} if problem_type=="classification" else {}),
    }
    model_json.write_text(json.dumps(report, indent=2))

    # 7) write brief markdown summary
    md_lines = [f"# Executive Summary – {ModelCls.__name__}", "", f"*{plain}*"]
    if feat_path_str:
        md_lines += ["", f"![Feature Importances]({feat_png})"]
    summary_md.write_text("\n\n".join(md_lines))

    print("✅ Model built and artefacts saved")
    return report

# ── force Pydantic to resolve any forward refs ─────────────────────────────
build_predictive_model.model_rebuild()
