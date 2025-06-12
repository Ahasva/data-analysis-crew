"""
Tool for model builder agent to properly create ML models.
Used in CrewAI pipelines to train multiple models, evaluate, and select the best one.

Returns both technical metrics and visualization plots in a structured format,
including model comparison, feature importances, and diagnostic charts.
"""
# ── src/data_analysis_crew/tools/build_predictive_model_tool.py ──────────────

# NOTE: matplotlib backend must be set before importing pyplot
# pylint: disable=wrong-import-position, wrong-import-order
import matplotlib
matplotlib.use("Agg") # noqa: E402
import matplotlib.pyplot as plt # noqa: E402

# --- Standard library imports ---
import json
from pathlib import Path
from typing import Any, Dict, Union

# --- Third-party packages ---
import shap
import pandas as pd
from crewai.tools import tool
from data_analysis_crew.crew import ModelOutput

# --- ML and plotting libraries ---
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, ConfusionMatrixDisplay,
    r2_score, mean_squared_error, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.inspection import permutation_importance
from sklearn.exceptions import NotFittedError

# --------------------------------------------------------------------------- #
#  Registry of available models (used in multi-model training)               #
# --------------------------------------------------------------------------- #
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "classification": {
        "random_forest": RandomForestClassifier,
        "logistic_reg": LogisticRegression,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "gbt": GradientBoostingClassifier,
    },
    "regression": {
        "random_forest": RandomForestRegressor,
        "linear_reg": LinearRegression,
        "svm": SVR,
        "gbt": GradientBoostingRegressor,
    },
}

# --------------------------------------------------------------------------- #
#  Hyperparameter grids for tuning                                           #
# --------------------------------------------------------------------------- #
_MODEL_PARAM_GRID = {
    "random_forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "logistic_reg": {"C": [0.1, 1.0, 10.0], "solver": ["liblinear"]},
    "svm": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]},
    "knn": {"n_neighbors": [3, 5, 7]},
    "gbt": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    "linear_reg": {},  # No tuning needed
}

# Root path for resolving relative paths consistently with Streamlit dashboard
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
#  Helper: path validation                                                   #
# --------------------------------------------------------------------------- #
def is_valid_path(path: Any) -> bool:
    """Check if the provided input is a usable file path."""
    return isinstance(path, str) and bool(path.strip())

# --------------------------------------------------------------------------- #
#  Helper: feature importances (fallbacks if needed)                         #
# --------------------------------------------------------------------------- #
def _get_importances(model, X_test, y_test, random_state: int = 42): # pylint: disable=invalid-name
    """Return (importances, label) or (None, None) if the model doesn't support it."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_, "Built-in"
    if hasattr(model, "coef_"):
        coef = model.coef_.ravel() if model.coef_.ndim == 2 else model.coef_
        return abs(coef), "Coefficient"
    try:
        res = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=random_state)
        return res.importances_mean, "Permutation"
    except (ValueError, NotFittedError, RuntimeError) as e:
        print(f"⚠️ Permutation importance failed: {e}")
        return None, None

# --------------------------------------------------------------------------- #
#  MAIN TOOL: build_predictive_model                                         #
# --------------------------------------------------------------------------- #
DataInput = Union[str, Path]
PathInput = Union[str, Path]

@tool("build_predictive_model")
def build_predictive_model(
    data: DataInput,
    *,
    target: str = "outcome",
    problem_type: Union[str, None] = None,
    out_dir: PathInput = "output",
    tuning: bool = True,
    explain: bool = True,
) -> dict:
    """
    Automatically train multiple ML models (with optional tuning) and select the best.

    Parameters
    ----------
    data : str or Path
        Path to cleaned CSV dataset.
    target : str
        Target variable name in dataset.
    problem_type : str, optional
        'classification' or 'regression'. Auto-inferred if None.
    out_dir : str or Path
        Output directory for plots and reports.
    tuning : bool
        If True, perform hyperparameter tuning via GridSearchCV.

    Returns
    -------
    dict : metadata about best model, metrics, and generated files
    """
    if not is_valid_path(data):
        raise ValueError("Invalid `data` path string provided")

    df = pd.read_csv(data)
    out_dir = Path(out_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if problem_type is None:
        problem_type = "classification" if (df[target].dtype == "O" or df[target].nunique() <= 10) else "regression"
    if problem_type not in _MODEL_REGISTRY:
        raise ValueError(f"Invalid problem_type: {problem_type}")

    X = df.drop(columns=[target]) # pylint: disable=invalid-name
    y = df[target]
    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, # pylint: disable=invalid-name
                                                        y,
                                                        stratify=stratify,
                                                        test_size=0.25,
                                                        random_state=42)

    scores = {}
    for candidate_name, CandidateCls in _MODEL_REGISTRY[problem_type].items(): # pylint: disable=invalid-name
        try:
            grid_params = _MODEL_PARAM_GRID.get(candidate_name, {})
            candidate = CandidateCls()

            if tuning and grid_params:
                grid = GridSearchCV(candidate, grid_params, cv=3,
                                    scoring=("f1_weighted" if problem_type == "classification" else "r2"),
                                    n_jobs=-1)
                grid.fit(X_train, y_train)
                y_pred = grid.predict(X_test)
                scores[candidate_name] = (
                    f1_score(y_test, y_pred, average="weighted")
                    if problem_type == "classification"
                    else r2_score(y_test, y_pred),
                    grid, y_pred,
                )
            else:
                candidate.fit(X_train, y_train)
                y_pred = candidate.predict(X_test)
                scores[candidate_name] = (
                    f1_score(y_test, y_pred, average="weighted")
                    if problem_type == "classification"
                    else r2_score(y_test, y_pred),
                    candidate, y_pred,
                )
        except (ValueError, RuntimeError) as e:
            print(f"⚠️ {candidate_name} failed during training: {e}")

    if not scores:
        raise RuntimeError("All models failed during training.")

    best_model_name = max(scores, key=lambda m: scores[m][0])
    _, best_model, y_pred = scores[best_model_name]
    model_type = type(best_model).__name__

        # ─── Plots ─────────────────────────────────────────────────────────────
    feat_path_str = None
    secondary_paths = []
    residuals_path = None

    # Feature importances
    importances, imp_label = _get_importances(best_model, X_test, y_test)
    if importances is not None:
        try:
            feat_png = plots_dir / "feature_importances.png"
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(X.columns, importances)
            ax.set_title("Feature Importances")
            ax.set_xlabel(f"Importance ({imp_label})")
            fig.tight_layout()
            fig.savefig(feat_png)
            plt.close(fig)
            feat_path_str = str(feat_png.relative_to(PROJECT_ROOT))
        except (ValueError, RuntimeError) as e:
            print(f"❌ Feature plot failed: {e}")

    # Diagnostic plots
    try:
        if problem_type == "classification":
            cm_png = plots_dir / "confusion_matrix.png"
            ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test).figure_.savefig(cm_png)
            secondary_paths.append(str(cm_png.relative_to(PROJECT_ROOT)))

            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                roc_png = plots_dir / "roc_curve.png"
                pr_png = plots_dir / "precision_recall.png"
                RocCurveDisplay.from_predictions(y_test, y_proba).figure_.savefig(roc_png)
                PrecisionRecallDisplay.from_predictions(y_test, y_proba).figure_.savefig(pr_png)
                secondary_paths.extend([
                    str(roc_png.relative_to(PROJECT_ROOT)),
                    str(pr_png.relative_to(PROJECT_ROOT)),
                ])
        else:
            resid_png = plots_dir / "residuals.png"
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Residuals")
            fig.tight_layout()
            fig.savefig(resid_png)
            plt.close(fig)
            secondary_paths.append(str(resid_png.relative_to(out_dir)))
    except (ValueError, RuntimeError) as e:
        print(f"❌ Diagnostic plots failed: {e}")

    # SHAP summary plot (optional, safe import)
    shap_path_str = None
    if explain:
        try:
            shap_path = plots_dir / "shap_summary.png"

            # ✅ Handle GridSearchCV wrapping
            model_for_shap = best_model.best_estimator_ if isinstance(best_model, GridSearchCV) else best_model
            explainer = shap.Explainer(model_for_shap, X_train)
            print(f"ℹ️ SHAP using {type(explainer).__name__}")
            shap_values = explainer(X_test)

            # Defensive fallback
            if shap_values is not None and hasattr(shap_values, "values"):
                fig = plt.figure(figsize=(8, 6))
                shap.plots.beeswarm(shap_values, show=False)
                plt.gcf().savefig(shap_path, bbox_inches="tight")
                plt.close()

                shap_path_str = str(shap_path.relative_to(PROJECT_ROOT))
                secondary_paths.append(shap_path_str)
            else:
                print("⚠️ SHAP values object was invalid or empty. Skipping SHAP plot.")
        except (ValueError, RuntimeError) as e:
            print(f"⚠️ SHAP explanation failed: {e}")

    # Bar chart of model scores
    score_vals = dict(sorted(
        ((k, round(v[0], 4)) for k, v in scores.items()),
        key=lambda item: item[1],
        reverse=True
    ))
    try:
        bar_path = plots_dir / "model_score_comparison.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(list(score_vals.keys()), list(score_vals.values()))
        ax.set_title("Model Score Comparison")
        ax.set_xlabel("Score")
        fig.tight_layout()
        fig.savefig(bar_path)
        plt.close(fig)
        secondary_paths.append(str(bar_path.relative_to(PROJECT_ROOT)))
    except (ValueError, RuntimeError, TypeError) as e:
        print(f"❌ Bar chart plot failed: {e}")

    # Residuals CSV export (for regression)
    if problem_type == "regression":
        try:
            residuals_path = (out_dir / "model-residuals.csv")
            pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred,
                "residual": y_test - y_pred
            }).to_csv(residuals_path, index=False)
        except (OSError, ValueError, KeyError) as e:
            print(f"❌ Could not write residuals.csv: {e}")
            residuals_path = None

    # ─── Final output ──────────────────────────────────────────────────────
    metrics = (
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        } if problem_type == "classification"
        else {
            "r2": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
        }
    )
    plain = (
        f"Accuracy {metrics['accuracy']:.2%}, F1 {metrics['f1']:.2%}"
        if problem_type == "classification"
        else f"R² {metrics['r2']:.3f}, MSE {metrics['mse']:.4f}"
    )

    report_json = out_dir / "model-report.json"
    report_md = out_dir / "technical-metrics.md"

    report = {
        "selected_model": best_model_name,
        "model_type": model_type,
        "tuning_enabled": tuning,
        "problem_type": problem_type,
        "target": target,
        "metrics": metrics,
        "plain_summary": plain,
        "all_model_scores": score_vals,
        "technical_summary_path": str(report_md.relative_to(out_dir)),
    }

    # ─── Consolidate optional outputs ───────────────
    output_files = {}

    if feat_path_str:
        output_files["feature_importance_path"] = feat_path_str

    if secondary_paths:
        output_files["secondary_plot_paths"] = secondary_paths
        if problem_type == "classification":
            for path in secondary_paths:
                if "confusion_matrix" in path:
                    output_files["confusion_matrix_path"] = path
                    break

    if shap_path_str:
        output_files["shap_summary_path"] = shap_path_str

    if residuals_path:
        output_files["residuals_path"] = str(residuals_path.relative_to(PROJECT_ROOT))

    report["outputs"] = output_files

    if isinstance(best_model, GridSearchCV):
        report["best_params"] = best_model.best_params_

    # Save full report as JSON
    report_json.write_text(json.dumps(report, indent=2))

    # Save markdown summary
    md_lines = [
        f"# Executive Summary – {model_type}",
        "",
        f"**Selected Model:** `{best_model_name}`",
        "",
        f"**Summary:** *{plain}*",
        "",
        "## Performance Metrics:",
    ]

    for k, v in metrics.items():
        md_lines.append(f"- {k.upper()}: {v:.4f}")

    md_lines.append("\n## Visual Summaries:")

    if feat_path_str:
        md_lines.append(f"- Feature Importances:\n  <img src=\"{feat_path_str}\" width=\"600\" />")

    conf_path = output_files.get("confusion_matrix_path")
    roc_path  = next((p for p in secondary_paths if "roc_curve" in p), None)
    pr_path   = next((p for p in secondary_paths if "precision_recall" in p), None)
    shap_path = output_files.get("shap_summary_path")

    if conf_path:
        md_lines.append(f"- Confusion Matrix:\n  <img src=\"{conf_path}\" width=\"600\" />")
    if roc_path:
        md_lines.append(f"- ROC Curve:\n  <img src=\"{roc_path}\" width=\"600\" />")
    if pr_path:
        md_lines.append(f"- Precision-Recall:\n  <img src=\"{pr_path}\" width=\"600\" />")
    if shap_path:
        md_lines.append(f"- SHAP Summary:\n  <img src=\"{shap_path}\" width=\"600\" />")

    # Best hyperparameters (if tuning was used)
    if isinstance(best_model, GridSearchCV):
        md_lines += [
            "\n## Best Hyperparameters:",
            "```json",
            json.dumps(best_model.best_params_, indent=2),
            "```",
        ]

    md_lines += ["", "## All Model Scores", "", "| Model | Score |", "|-------|-------|"]
    for name, score in score_vals.items():
        md_lines.append(f"| {name} | {score:.4f} |")

    report_md.write_text("\n".join(md_lines))

    print(f"✅ Best model selected: {best_model_name}")

    return ModelOutput(
        model_type=model_type,
        problem_type=problem_type,
        target=target,
        metrics=metrics,
        plain_summary=plain,
        feature_importance_path=feat_path_str,
        secondary_plot_paths=secondary_paths if secondary_paths else [],
        confusion_matrix_path=output_files.get("confusion_matrix_path")
    )
