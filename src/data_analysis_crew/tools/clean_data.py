import pandas as pd
from typing import Optional
from crewai.tools import tool
from data_analysis_crew.schemas import CleanedDataOutput
from data_analysis_crew.utils.utils import to_posix_relative_path
from data_analysis_crew.utils.project_root import resolve_path, get_project_root


@tool("load_or_clean")
def load_or_clean(
    raw_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    cleaned_path: Optional[str] = None,
) -> CleanedDataOutput:
    """
    If `cleaned_path` exists, load and return metadata.
    Otherwise:
      - Read from `raw_path` or `dataset_path`
      - Normalize column names (strip, lowercase, replace spaces with underscores)
      - Save cleaned CSV to `cleaned_path`
      - Return structured metadata using CleanedDataOutput schema.
    """
    source = raw_path or dataset_path
    if not source:
        raise ValueError("Either `raw_path` or `dataset_path` must be provided.")
    if not cleaned_path:
        raise ValueError("Missing `cleaned_path` argument.")

    raw_file = resolve_path(source)
    clean_file = resolve_path(cleaned_path)

    print(f"📂 Checking for existing cleaned file: {clean_file}")

    if clean_file.exists():
        df = pd.read_csv(clean_file)
        print(f"✅ Loaded existing cleaned file: {clean_file}")
    else:
        print(f"🧼 Cleaning raw file: {raw_file}")
        df = pd.read_csv(raw_file)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        clean_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(clean_file, index=False)
        print(f"✅ Saved new cleaned file: {clean_file}")

    # Assemble output
    cleaned_posix = to_posix_relative_path(clean_file.resolve(), get_project_root())
    return CleanedDataOutput(
        cleaned_path=cleaned_posix,
        final_features=df.columns.tolist(),
        categorical_features=df.select_dtypes(include=['object', 'category']).columns.tolist(),
        numeric_features=df.select_dtypes(include='number').columns.tolist(),
        dropped_columns=[],  # Extend logic in the future
        imputation_summary=None  # Extend logic in the future
    )
