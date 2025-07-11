# ── src/data_analysis_crew/schemas.py ──
from typing import Optional, List, Dict, Tuple, Literal
from pydantic import BaseModel, Field

# ======= OUTPUT SCHEMAS =======
class LoadDataOutput(BaseModel):
    dataset_path: str = Field(description="Path to the loaded dataset")
    shape: Tuple[int, int] = Field(
        description="Shape of the dataset (rows, columns)",
        json_schema_extra={
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
    )
    columns: List[str] = Field(
        description="List of dataset columns",
        json_schema_extra={"items": {"type": "string"}}
    )
    dtype_map: Optional[Dict[str, str]] = Field(
        default=None,
        description="Data type for each column"
    )
    missing_values: Optional[Dict[str, int]] = Field(
        default=None,
        description="Count of missing values per column"
    )

class CleanedDataOutput(BaseModel):
    cleaned_path: str = Field(description="Path to the cleaned dataset file")
    final_features: List[str] = Field(
        description="List of features retained after cleaning"
    )
    categorical_features: List[str] = Field(
        description="List of identified categorical features"
    )
    numeric_features: List[str] = Field(
        description="List of identified numerical features"
    )
    dropped_columns: List[str] = Field(
        description="List of columns dropped during cleaning"
    )
    imputation_summary: Optional[Dict[str, str]] = Field(
        default=None,
        description="Summary of how missing values were handled"
    )
    summary_markdown: Optional[str] = Field(
        default=None,
        description="Markdown summary of cleaning steps, column transformations, and missing value handling"
    )
    summary_path_json: Optional[str] = Field(
        default=None,
        description="Path to the structured JSON file containing the cleaning summary metadata"
    )
    summary_path_md: Optional[str] = Field(
        default=None,
        description="Path to the Markdown file describing the cleaning process"
    )

class FeatureCorrelation(BaseModel):
    feature: str = Field(description="Feature name")
    correlation: float = Field(description="Correlation coefficient with the target")

class ExplorationOutput(BaseModel):
    plot_paths: List[str] = Field(
        description="Paths to saved plots from data exploration"
    )
    top_correlations: List[FeatureCorrelation] = Field(
        description="Top correlated features with the target"
    )
    anomalies: Optional[List[Dict[str, float]]] = Field(
        default_factory=list,
        description="List of potential data anomalies"
    )
    statistical_notes: str = Field(description="Narrative summary of statistical insights")

class FeatureSelectionOutput(BaseModel):
    problem_type: Literal["classification", "regression"] = Field(
        description="Inferred ML problem type"
    )
    top_features: List[str] = Field(
        description="List of selected top features"
    )
    reasoning: str = Field(description="Explanation for selected features and problem type")

class ModelOutput(BaseModel):
    model_type: str = Field(
        description="Sklearn class name of the trained model (e.g. RandomForestClassifier, SVR)."
    )
    problem_type: Literal["classification", "regression"] = Field(
        description="Problem formulation inferred by the pipeline."
    )
    target: str = Field(
        description="Name of the target column that was predicted."
    )
    metrics: Optional[Dict[str, float]] = Field(
        description=(
            "Primary evaluation metrics. "
            "For classification: {'accuracy','f1'}; "
            "for regression: {'r2','mse'}."
        )
    )
    plain_summary: str = Field(
        description="Short one-liner summarising the metrics (shown on the dashboard card)."
    )
    feature_importance_path: Optional[str] = Field(
        default=None,
        description="Relative path to feature-importance PNG (may be None if not supported)."
    )
    secondary_plot_paths: Optional[List[str]] = Field(
        default=None,
        description="List of paths to additional plots like ROC, PR, residuals, etc."
    )
    confusion_matrix_path: Optional[str] = Field(
        default=None,
        description="(DEPRECATED) alias of `secondary_plot_paths` when `problem_type=='classification'`."
    )
    residuals_plot_path: Optional[str] = Field(
        default=None,
        description="Path to the residuals scatter plot for regression models."
    )

class SummaryReportOutput(BaseModel):
    summary: str = Field(
        description="A short executive summary of the findings, highlighting the key conclusions from the analysis and modeling."
    )
    insights: List[str] = Field(
        description="A list of 3–5 bullet-point insights derived from the data exploration and modeling results."
    )
    recommendation: str = Field(
        description="A clear, actionable recommendation based on the model’s performance and insights derived from the data."
    )
    metrics: Dict[str, float] = Field(
        description="A dictionary of the key model performance metrics, such as accuracy, F1 score, R2, or MSE."
    )
    image_paths: List[str] = Field(
        description="List of file paths to images (e.g. PNGs) generated during analysis or model training, such as feature importance or ROC curves."
    )