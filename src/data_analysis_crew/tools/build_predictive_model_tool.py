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
from sklearn.exceptions import NotFittedError

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
#  Path validation helper                                                     #
# --------------------------------------------------------------------------- #
def is_valid_path(path: Any) -> bool:
    """Return True if `path` is a non-empty string with visible characters."""
    return isinstance(path, str) and bool(path.strip())

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
    try:
        res = permutation_importance(
            model, X_test, y_test, n_repeats=15, random_state=random_state
        )
        return res.importances_mean, "Permutation"
    except (ValueError, NotFittedError, RuntimeError) as e:
        print(f"⚠️ Permutation importance failed: {e}")
        return None, None

# --------------------------------------------------------------------------- #
#  Main tool                                                                  #
# --------------------------------------------------------------------------- #
@tool("build_predictive_model")
def build_predictive_model(
    data: pd.DataFrame | str | Path,
    *,
    target: str = "outcome",
    problem_type: str | None = None,
    model_name: str | None = None,
    out_dir: str | Path = "output",
    **model_kwargs,
) -> dict:
    """
    Train a model and save:

      • output/model-report.json  
      • output/final-insight-summary.md  
      • output/plots/feature_importances.png (if available)  
      • output/plots/confusion_matrix.png or residuals.png

    Parameters
    ----------
    data         : DataFrame or str/Path to a CSV containing the target.
    target       : Target column name (default: "outcome").
    problem_type : "classification" | "regression" | None (auto-infer).
    model_name   : See registry above (e.g. "svm", "gbt").
    out_dir      : Root folder for artefacts.
    **model_kwargs : Passed straight to the model constructor.

    Note
    ----
    All paths saved in the output dict (JSON) are **relative to out_dir**.
    This ensures Markdown and dashboard embedding works consistently.
    - Markdown: ![Feature](plots/feature_importances.png)
    - JSON: "feature_importance_path": "plots/feature_importances.png"

    Returns
    -------
    dict : Same payload written to *model-report.json*
    """
    # 0) Ingest data
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("`data` must be a pandas DataFrame or a CSV path/string")

    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_json = out_dir / "model-report.json"
    summary_md = out_dir / "technical-metrics.md"

    # 1) Infer problem type
    if problem_type is None:
        problem_type = (
            "classification"
            if (df[target].dtype == "O" or df[target].nunique() <= 10)
            else "regression"
        )

    if problem_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown problem_type {problem_type!r}")

    # 2) Choose model
    if model_name is None:
        model_name = "random_forest"

    if model_name not in _MODEL_REGISTRY[problem_type]:
        valid = ", ".join(_MODEL_REGISTRY[problem_type])
        raise ValueError(f"{model_name!r} invalid for {problem_type}. Valid: {valid}")

    # 3) Train-test split
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    # 4) Train model
    ModelCls = _MODEL_REGISTRY[problem_type][model_name]
    default_kwargs = {"random_state": 42} if "random_state" in ModelCls().__dict__ else {}
    model = ModelCls(**{**default_kwargs, **model_kwargs})
    model.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test)
    metrics = (
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }
        if problem_type == "classification"
        else {
            "r2": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
        }
    )

    # 6) Save plots
    feat_png = plots_dir / "feature_importances.png"
    secondary_png = plots_dir / (
        "confusion_matrix.png" if problem_type == "classification" else "residuals.png"
    )

    # Feature importances
    importances, imp_label = _get_importances(model, X_test, y_test)
    feat_path_str = None
    if importances is not None:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(X.columns, importances)
            ax.set_xlabel(f"Importance ({imp_label})")
            ax.set_title("Feature Importances")
            fig.tight_layout()
            fig.savefig(feat_png)
            plt.close(fig)
            feat_path_str = str(feat_png.relative_to(out_dir))
            if is_valid_path(feat_path_str):
                print(f"✅ Saved feature importances to {feat_path_str}")
        except (ValueError, IndexError, RuntimeError) as e:
            print(f"❌ Failed to plot feature importances: {e}")
            feat_path_str = None
    else:
        print("⚠️ No feature importances available — skipping plot.")

    # Secondary plot
    try:
        if problem_type == "classification":
            disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
            disp.figure_.savefig(secondary_png)
            plt.close(disp.figure_)
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Residual Plot")
            fig.tight_layout()
            fig.savefig(secondary_png)
            plt.close(fig)
        secondary_path_str = str(secondary_png.relative_to(out_dir))
    except (ValueError, RuntimeError) as e:
        print(f"❌ Failed to save secondary plot: {e}")
        secondary_path_str = None

    # 7) Assemble report
    plain = (
        f"Accuracy {metrics['accuracy']:.2%}, F1 {metrics['f1']:.2%}"
        if problem_type == "classification"
        else f"R² {metrics['r2']:.3f}, MSE {metrics['mse']:.4f}"
    )
    report = {
        "model_type": ModelCls.__name__,
        "problem_type": problem_type,
        "target": target,
        "metrics": metrics,
        "plain_summary": plain,
        "technical_summary_path": str(summary_md.relative_to(out_dir)),
    }

    if is_valid_path(feat_path_str):
        report["feature_importance_path"] = feat_path_str
    if is_valid_path(secondary_path_str):
        report["secondary_plot_path"] = secondary_path_str
    if problem_type == "classification" and is_valid_path(secondary_path_str):
        report["confusion_matrix_path"] = secondary_path_str

    model_json.write_text(json.dumps(report, indent=2))

    # 8) Write technical markdown
    md_lines = [f"# Executive Summary – {ModelCls.__name__}", "", f"*{plain}*"]
    if is_valid_path(feat_path_str):
        md_lines += ["", f"![Feature Importances]({feat_path_str})"]
    summary_md.write_text("\n\n".join(md_lines))
    print(f"📄 Technical summary written to {summary_md}")

    print("✅ Model built and artefacts saved")
    return report

# ── force Pydantic to resolve any forward refs ─────────────────────────────
build_predictive_model.model_rebuild()
