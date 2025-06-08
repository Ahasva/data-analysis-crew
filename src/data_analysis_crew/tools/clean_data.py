# --- clean_data.py -----------------------------------------------------------
from pathlib import Path
import pandas as pd

RAW_PATH     = Path("knowledge/diabetes.csv")
CLEAN_PATH   = Path("knowledge/diabetes_cleaned.csv")

# ---------------------------------------------------------------------
def load_or_clean(raw_path: Path, cleaned_path: Path) -> pd.DataFrame:
    """
    – If cleaned_path exists → load & return (no re-processing).
    – Else → read raw_path, normalise columns, clean, save & return.
    """
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        print(f"✔ Re-using cached clean file: {cleaned_path}")
        return df

    # ---------- minimal cleaning example ----------
    df = pd.read_csv(raw_path)
    df.columns = (
        df.columns.str.strip()      # remove leading/trailing spaces
                 .str.lower()
                 .str.replace(" ", "_")
    )
    # (Add your imputation / dropping / dtype fixes here)
    df.to_csv(cleaned_path, index=False)
    print(f"💾 Saved cleaned file to: {cleaned_path}")
    return df
# ---------------------------------------------------------------------

df_load_or_clean = load_or_clean(RAW_PATH, CLEAN_PATH)

# Collect the metadata your Pydantic model expects
output = {
    "cleaned_path": str(CLEAN_PATH),
    "final_features": df_load_or_clean.columns.tolist(),
    "categorical_features": [],   #  <- fill if you detect them
    "numeric_features": df_load_or_clean.select_dtypes("number").columns.tolist(),
    "dropped_columns": [],        #  <- add any you removed
    "imputation_summary": None    #  <- or a dict of {col:method}
}
print(output["cleaned_path"])      # <-- **Option 1: always print for the tool**
print(output)                      # optional: shows the rest for logs