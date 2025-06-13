from pathlib import Path
import pandas as pd
from crewai.tools import tool

@tool("load_or_clean")
def load_or_clean(
    raw_path: str = None,
    dataset_path: str = None,
    cleaned_path: str = None,
) -> dict:
    """
    – If `cleaned_path` exists → load & return (no re-processing).
    – Else → read from `raw_path` or `dataset_path`, normalize column names, save & return.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    source = raw_path or dataset_path
    if source is None:
        raise ValueError("Either `raw_path` or `dataset_path` must be provided.")

    raw = Path(source)
    clean = Path(cleaned_path)
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    if not clean.is_absolute():
        clean = PROJECT_ROOT / clean

    print(f"🚧 DEBUG load_or_clean → raw_file   = {raw}")
    print(f"🚧 DEBUG load_or_clean → clean_file = {clean}")

    if clean.exists():
        print(f"📂 Loading existing cleaned file from: {clean}")
        df = pd.read_csv(clean)
        print(f"✅ Loaded {len(df)} rows from cleaned file.")
    else:
        print(f"🧹 Cleaning raw file: {raw}")
        df = pd.read_csv(raw)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        df.to_csv(clean, index=False)
        print(f"✅ Loaded {len(df)} rows from raw file and saved cleaned version to: {clean}")

    return {
        "cleaned_path": str(clean),
        "final_features": df.columns.tolist(),
        "categorical_features": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "numeric_features": df.select_dtypes(include="number").columns.tolist(),
        "dropped_columns": [],  # Future improvement: track any removed cols
        "imputation_summary": None  # Future improvement: fill method, stats
    }
