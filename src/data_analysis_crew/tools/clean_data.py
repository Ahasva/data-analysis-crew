import os
import json
import pandas as pd
from crewai.tools import tool
from data_analysis_crew.schemas import CleanedDataOutput 

@tool("clean_data_tool")
def clean_data_tool(raw_path: str, cleaned_path: str, summary_path_json: str, summary_path_md: str) -> CleanedDataOutput:
    """
    Cleans the raw dataset and outputs:
    - a cleaned CSV file
    - a JSON metadata summary
    - a Markdown summary report
    """

    df = pd.read_csv(raw_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.dropna()

    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df.to_csv(cleaned_path, index=False)

    categorical = list(df.select_dtypes(include=["object", "category"]).columns)
    numeric = list(df.select_dtypes(include=["number"]).columns)

    summary_md = (
        f"## Data Cleaning Summary\n"
        f"- **Dropped Columns:** None\n"
        f"- **Categorical Features:** {', '.join(categorical)}\n"
        f"- **Numeric Features:** {', '.join(numeric)}\n"
    )

    # Save JSON
    summary_json = {
        "cleaned_path": cleaned_path,
        "final_features": list(df.columns),
        "categorical_features": categorical,
        "numeric_features": numeric,
        "dropped_columns": [],
        "imputation_summary": {"strategy": "Dropped rows with missing values"},
        "summary_markdown": summary_md,
        "summary_path_json": summary_path_json,
        "summary_path_md": summary_path_md
    }

    os.makedirs(os.path.dirname(summary_path_json), exist_ok=True)
    with open(summary_path_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # Save Markdown
    os.makedirs(os.path.dirname(summary_path_md), exist_ok=True)
    with open(summary_path_md, "w", encoding="utf-8") as f:
        f.write(summary_md)

    #return summary_json # needs typehint in line def clean_data_tool(...) -> dict:
    # ✅ Explicit return as typed Pydantic model
    return CleanedDataOutput(**summary_json)
