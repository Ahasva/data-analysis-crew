# ── src/data_analysis_crew/tools/build_predictive_model_tool.py ──────────────
"""
Tool for model builder agent to properly create ML models.
Used in CrewAI pipelines to train multiple models, evaluate, and select the best one.

Returns both technical metrics and visualization plots in a structured format,
including model comparison, feature importances, and diagnostic charts.
"""

# ────────────────────────── Imports ────────────────────────── #
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import json
from pathlib import Path
from typing import Any, Dict, Union
from pydantic import ValidationError
import shap
import pandas as pd
from pandas.errors import EmptyDataError
from crewai.tools import tool
from data_analysis_crew.schemas import ModelOutput
from data_analysis_crew.utils.utils import to_posix_relative_path
from data_analysis_crew.utils.project_root import resolve_path, get_project_root
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

# ────────────────────────── Configs ────────────────────────── #
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "classification": {
        "random_forest": RandomForestClassifier,
        "logistic_reg":  LogisticRegression,
        "svm":           SVC,
        "knn":           KNeighborsClassifier,
        "gbt":           GradientBoostingClassifier,
    },
    "regression": {
        "random_forest": RandomForestRegressor,
        "linear_reg":    LinearRegression,
        "svm":           SVR,
        "gbt":           GradientBoostingRegressor,
    },
}

_MODEL_PARAM_GRID = {
    "random_forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "logistic_reg":  {"C": [0.1, 1.0, 10.0], "solver": ["liblinear"]},
    "svm":           {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]},
    "knn":           {"n_neighbors": [3, 5, 7]},
    "gbt":           {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    "linear_reg":    {},
}


def is_valid_path(path: Any) -> bool:
    return isinstance(path, str) and bool(path.strip())

def _get_importances(model, X_test, y_test, random_state: int = 42):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_, "Built-in"
    if hasattr(model, "coef_"):
        coef = model.coef_.ravel() if model.coef_.ndim == 2 else model.coef_
        return abs(coef), "Coefficient"
    try:
        res = permutation_importance(model, X_test, y_test,
                                     n_repeats=15, random_state=random_state)
        return res.importances_mean, "Permutation"
    except (ValueError, NotFittedError, RuntimeError) as e:
        print(f"⚠️ Permutation importance failed: {e}")
        return None, None

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
    Build, tune, and evaluate machine learning models from a dataset.
    Generates plots, saves outputs, and returns structured metrics.
    """
    if not is_valid_path(data):
        raise ValueError("Invalid `data` path string provided")
    df = pd.read_csv(resolve_path(data))

    project_root = get_project_root()

    out_dir = resolve_path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Writing plots to {plots_dir}")

    if problem_type is None:
        problem_type = (
            "classification" if df[target].dtype == "O" or df[target].nunique() <= 10
            else "regression"
        )
    if problem_type not in _MODEL_REGISTRY:
        raise ValueError(f"Invalid problem_type: {problem_type}")

    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=stratify, test_size=0.25, random_state=42
    )

    scores: Dict[str, tuple] = {}
    for name, Cls in _MODEL_REGISTRY[problem_type].items():
        try:
            params = _MODEL_PARAM_GRID.get(name, {})
            model = Cls()
            if tuning and params:
                grid = GridSearchCV(model, params, cv=3,
                    scoring=("f1_weighted" if problem_type == "classification" else "r2"),
                    n_jobs=-1)
                grid.fit(X_train, y_train)
                y_pred = grid.predict(X_test)
                scores[name] = (
                    f1_score(y_test, y_pred, average="weighted")
                    if problem_type == "classification"
                    else r2_score(y_test, y_pred),
                    grid, y_pred)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores[name] = (
                    f1_score(y_test, y_pred, average="weighted")
                    if problem_type == "classification"
                    else r2_score(y_test, y_pred),
                    model, y_pred)
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"⚠️ {name} failed during training: {e}")

    if not scores:
        raise RuntimeError("All models failed during training.")

    best_name = max(scores, key=lambda k: scores[k][0])
    _, best_model, y_pred = scores[best_name]
    model_type = type(best_model).__name__

    feat_path_str = None
    secondary_paths: list[str] = []
    residuals_path = None

    importances, imp_label = _get_importances(best_model, X_test, y_test)
    if importances is not None:
        feat_png = plots_dir / "feature_importance.png"
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(X.columns, importances)
        ax.set_title("Feature Importances")
        ax.set_xlabel(f"Importance ({imp_label})")
        fig.tight_layout()
        fig.savefig(feat_png)
        plt.close(fig)
        feat_path_str = to_posix_relative_path(feat_png, project_root)

    try:
        if problem_type == "classification":
            cm_png = plots_dir / "confusion_matrix.png"
            disp = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
            disp.figure_.savefig(cm_png)
            plt.close(disp.figure_)
            secondary_paths.append(to_posix_relative_path(cm_png, project_root))

            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]

                roc_png = plots_dir / "roc_curve.png"
                RocCurveDisplay.from_predictions(y_test, y_proba).figure_.savefig(roc_png)
                plt.close()
                secondary_paths.append(to_posix_relative_path(roc_png, project_root))

                pr_png = plots_dir / "precision_recall.png"
                PrecisionRecallDisplay.from_predictions(y_test, y_proba).figure_.savefig(pr_png)
                plt.close()
                secondary_paths.append(to_posix_relative_path(pr_png, project_root))
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
            residuals_plot_path = to_posix_relative_path(resid_png, project_root)
            secondary_paths.append(residuals_plot_path)
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"❌ Diagnostic plots failed: {e}")

    shap_path_str = None
    if explain:
        try:
            shap_file = plots_dir / "shap_summary.png"
            explainer = shap.Explainer(
                best_model.best_estimator_ if isinstance(best_model, GridSearchCV) else best_model,
                X_train
            )
            shap_vals = explainer(X_test)
            if hasattr(shap_vals, "values"):
                plt.figure(figsize=(8,6))
                shap.plots.beeswarm(shap_vals, show=False)
                plt.gcf().savefig(shap_file, bbox_inches="tight")
                plt.close()
                shap_path_str = to_posix_relative_path(shap_file, project_root)
                secondary_paths.append(shap_path_str)
        except (ValueError, RuntimeError, AttributeError) as e:
            print(f"⚠️ SHAP explanation failed: {e}")

    try:
        bar_png = plots_dir / "model_score_comparison.png"
        fig, ax = plt.subplots(figsize=(6,4))
        score_vals = {
            k: round(v[0], 4) if isinstance(v, tuple) and isinstance(v[0], (int, float)) else 0.0
            for k, v in scores.items()
        }
        ax.barh(list(score_vals.keys()), list(score_vals.values()))
        ax.set_title("Model Score Comparison")
        ax.set_xlabel("Score")
        fig.tight_layout()
        fig.savefig(bar_png)
        plt.close(fig)
        secondary_paths.append(to_posix_relative_path(bar_png, project_root))
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"❌ Bar chart plot failed: {e}")

    if problem_type == "regression":
        try:
            residuals_path = out_dir / "model-residuals.csv"
            pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred,
                "residual": y_test - y_pred
            }).to_csv(residuals_path, index=False)
        except (PermissionError, OSError, EmptyDataError, TypeError) as e:
            print(f"❌ Could not write residuals.csv: {e}")

    metrics = (
        {"accuracy": accuracy_score(y_test, y_pred),
         "f1":       f1_score(y_test, y_pred, average="weighted")}
        if problem_type == "classification"
        else {"r2": r2_score(y_test, y_pred), "mse": mean_squared_error(y_test, y_pred)}
    )
    plain = (
        f"Accuracy {metrics['accuracy']:.2%}, F1 {metrics['f1']:.2%}"
        if problem_type == "classification"
        else f"R² {metrics['r2']:.3f}, MSE {metrics['mse']:.4f}"
    )

    report_json = out_dir / "model-report.json"
    report_md = out_dir / "technical-metrics.md"

    outputs = {}
    if feat_path_str:
        outputs["feature_importance_path"] = feat_path_str
    if secondary_paths:
        outputs["secondary_plot_paths"] = secondary_paths
        if problem_type == "classification":
            cm = next((p for p in secondary_paths if "confusion_matrix" in p), None)
            if cm:
                outputs["confusion_matrix_path"] = cm
    if shap_path_str:
        outputs["shap_summary_path"] = shap_path_str
    if residuals_path:
        outputs["residuals_path"] = to_posix_relative_path(residuals_path, project_root)
    residuals_plot_path = next((p for p in secondary_paths if "residuals.png" in p), None)
    if residuals_plot_path:
        outputs["residuals_plot_path"] = residuals_plot_path

    report = {
        "selected_model": best_name,
        "model_type": model_type,
        "tuning_enabled": tuning,
        "problem_type": problem_type,
        "target": target,
        "metrics": metrics,
        "plain_summary": plain,
        "all_model_scores": score_vals,
        "technical_summary_path": to_posix_relative_path(report_md, project_root),
        "outputs": outputs,
    }

    if isinstance(best_model, GridSearchCV):
        report["best_params"] = best_model.best_params_

    report_json.write_text(json.dumps(report, indent=2))

    md = [
        f"# Executive Summary – {model_type}",
        "",
        f"**Selected Model:** `{best_name}`",
        "",
        f"**Summary:** *{plain}*",
        "",
        "## Performance Metrics:",
    ]
    for k, v in metrics.items():
        md.append(f"- {k.upper()}: {v:.4f}")
    md.append("\n## Visual Summaries:")
    if feat_path_str:
        md.append(f"- Feature Importances:\n  <img src=\"{feat_path_str}\" width=\"600\"/>")
    if outputs.get("confusion_matrix_path"):
        md.append(f"- Confusion Matrix:\n  <img src=\"{outputs['confusion_matrix_path']}\" width=\"600\"/>")
    for plot in ["roc_curve", "precision_recall", "shap_summary", "residuals"]:
        path = next((p for p in secondary_paths if plot in p), None)
        if path:
            md.append(f"- {plot.replace('_', ' ').title()}:\n  <img src=\"{path}\" width=\"600\"/>")
    if isinstance(best_model, GridSearchCV):
        md += ["\n## Best Hyperparameters:", "```json", json.dumps(best_model.best_params_, indent=2), "```"]
    md += ["", "## All Model Scores", "", "| Model | Score |", "|-------|-------|"]
    for nm, sc in score_vals.items():
        md.append(f"| {nm} | {sc:.4f} |")
    report_md.write_text("\n".join(md))

    print(f"✅ Best model selected: {best_name}")

    try:
        return ModelOutput(
            model_type=model_type,
            problem_type=problem_type,
            target=target,
            metrics=metrics,
            plain_summary=plain,
            feature_importance_path=feat_path_str,
            secondary_plot_paths=secondary_paths,
            confusion_matrix_path=outputs.get("confusion_matrix_path"),
            residuals_plot_path=outputs.get("residuals_plot_path")
        )
    except ValidationError as ve:
        print("❌ ModelOutput validation failed:")
        print(ve.json(indent=2))
        raise
