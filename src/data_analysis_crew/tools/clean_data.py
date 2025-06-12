"""
Tool for cleaning the dataset
"""
# --- clean_data.py -----------------------------------------------------------
from pathlib import Path
import pandas as pd
from crewai.tools import tool

# ---------------------------------------------------------------------
@tool("load_or_clean")
def load_or_clean(
    raw_path: str = "knowledge/diabetes.csv",
    cleaned_path: str = None,
) -> dict:
    """
    – If cleaned_path exists → load & return (no re-processing).
    – Else → read raw_path, normalize column names, save & return.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    if cleaned_path is None:
        cleaned_path = "knowledge/diabetes_cleaned.csv"

    raw   = PROJECT_ROOT / raw_path
    clean = PROJECT_ROOT / cleaned_path

    # load or clean
    if clean.exists():
        df = pd.read_csv(clean)
    else:
        df = pd.read_csv(raw)
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
        )
        df.to_csv(clean, index=False)

    return {
        "cleaned_path"        : str(clean),
        "final_features"      : df.columns.tolist(),
        "categorical_features": [],
        "numeric_features"    : df.select_dtypes("number").columns.tolist(),
        "dropped_columns"     : [],
        "imputation_summary"  : None
    }
