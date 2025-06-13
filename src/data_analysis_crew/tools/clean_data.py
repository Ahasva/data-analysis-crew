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
    raw_path: str,
    cleaned_path: str,
) -> dict:
    """
    – If `cleaned_path` exists → load & return (no re-processing).
    – Else → read `raw_path`, normalize column names, save & return.
    """
    # Determine project root (3 levels up)
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Build absolute paths from inputs
    raw = Path(raw_path)
    clean = Path(cleaned_path)
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    if not clean.is_absolute():
        clean = PROJECT_ROOT / clean

    # Load existing cleaned file, or process raw
    if clean.exists():
        df = pd.read_csv(clean)
    else:
        df = pd.read_csv(raw)
        # Normalize column names to lower_snake_case
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
        )
        df.to_csv(clean, index=False)

    # Return rich metadata for downstream tasks
    return {
        "cleaned_path": str(clean),
        "final_features": df.columns.tolist(),
        "categorical_features": [],
        "numeric_features": df.select_dtypes(include="number").columns.tolist(),
        "dropped_columns": [],
        "imputation_summary": None
    }
