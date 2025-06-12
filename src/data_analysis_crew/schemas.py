# ── src/data_analysis_crew/schemas.py ──
from typing import Optional, List, Dict, Tuple, Literal
from pydantic import BaseModel, Field


# ======= OUTPUT SCHEMAS =======
class LoadDataOutput(BaseModel):
    dataset_path: str = Field(description="Path to the loaded dataset")
    # ⚠️  Fix: add `items` so OpenAI function schema is valid
    shape: Tuple[int, int] = Field(
        description="Shape of the dataset (rows, columns)",
        json_schema_extra={
            "items": {"type": "integer"},  # <— required by OpenAI
            "minItems": 2,
            "maxItems": 2,
        },
    )
    columns: List[str] = Field(description="List of dataset columns")
    dtype_map: Optional[Dict[str, str]] = Field(default=None, description="Data type for each column")
    missing_values: Optional[Dict[str, int]] = Field(default=None, description="Count of missing values per column")

class CleanedDataOutput(BaseModel):
    cleaned_path: str = Field(description="Path to the cleaned dataset file")
    final_features: List[str] = Field(description="List of features retained after cleaning")
    categorical_features: List[str] = Field(description="List of identified categorical features")
    numeric_features: List[str] = Field(description="List of identified numerical features")
    dropped_columns: List[str] = Field(description="List of columns dropped during cleaning")
    imputation_summary: Optional[Dict[str, str]] = Field(default=None, description="Summary of how missing values were handled")

class FeatureCorrelation(BaseModel):
    feature: str = Field(description="Feature name")
    correlation: float = Field(description="Correlation coefficient with the target")

class ExplorationOutput(BaseModel):
    plot_paths: List[str] = Field(description="Paths to saved plots from data exploration")
    top_correlations: List[FeatureCorrelation] = Field(description="Top correlated features with the target")
    anomalies: List[str] = Field(description="List of potential data anomalies")
    statistical_notes: str = Field(description="Narrative summary of statistical insights")

class FeatureSelectionOutput(BaseModel):
    problem_type: Literal["classification", "regression"] = Field(description="Inferred ML problem type")
    top_features: List[str] = Field(description="List of selected top features")
    reasoning: str = Field(description="Explanation for selected features and problem type")

class ModelOutput(BaseModel):
    # ── core info ───────────────────────────────────────────────────────
    model_type: str = Field(
        description="Sklearn class name of the trained model (e.g. RandomForestClassifier, SVR)."
    )
    problem_type: Literal["classification", "regression"] = Field(
        description="Problem formulation inferred by the pipeline."
    )
    target: str = Field(
        description="Name of the target column that was predicted."
    )

    # ── evaluation ──────────────────────────────────────────────────────
    metrics: Dict[str, float] = Field(
        description="Primary evaluation metrics. "
                    "For classification: {'accuracy','f1'}; "
                    "for regression: {'r2','mse'}."
    )
    plain_summary: str = Field(
        description="Short one-liner summarising the metrics (shown on the dashboard card)."
    )

    # ── artefacts (optional because some models lack importances) ───────
    feature_importance_path: Optional[str] = Field(
        default=None,
        description="Relative path to feature-importance PNG "
                    "(may be None if not supported)."
    )
    secondary_plot_paths: Optional[List[str]] = Field(
    default=None,
    description="List of paths to additional plots like ROC, PR, residuals, etc."
)

    # ── legacy alias for confusion matrix ───────────────────────────────
    confusion_matrix_path: Optional[str] = Field(
        default=None,
        description="(DEPRECATED) alias of `secondary_plot_path` when "
                    "`problem_type=='classification'`."
    )
