from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import tool
from data_analysis_crew.utils.utils import to_posix_relative_path

@tool("explore_data")
def explore_data(
    cleaned_path: str,
    plot_path: str
) -> dict:
    """
    Perform exploratory data analysis (EDA) on a cleaned dataset.

    - Reads the dataset from `cleaned_path`.
    - Generates histograms for all numeric features.
    - Creates a correlation heatmap.
    - Identifies top 3 features most correlated with the target column 'outcome'.
    - Detects outliers based on IQR.

    Saves plots under `plot_path` and returns:
      - POSIX-style paths to saved plots
      - Statistical notes
      - Top correlated features
      - Anomalies (outliers)
    """
    # Resolve project root and input/output paths
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    data_file = PROJECT_ROOT / cleaned_path
    plots_dir = PROJECT_ROOT / plot_path
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    df = pd.read_csv(data_file)
    plot_paths = []

    # 1. Histograms for numeric columns
    for col in df.select_dtypes(include="number").columns:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        out_file = plots_dir / f"distribution_{col}.png"
        fig.savefig(out_file)
        plt.close(fig)
        plot_paths.append(to_posix_relative_path(out_file, PROJECT_ROOT))

    # 2. Correlation heatmap
    corr_matrix = df.corr(numeric_only=True)
    heatmap_file = plots_dir / "correlation_heatmap.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="YlOrRd")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_file)
    plt.close()
    plot_paths.append(to_posix_relative_path(heatmap_file, PROJECT_ROOT))

    # 3. Top 3 correlations with 'outcome'
    if 'outcome' in corr_matrix.columns:
        top_corr = (
            corr_matrix['outcome']
            .drop('outcome')
            .abs()
            .sort_values(ascending=False)
            .head(3)
        )
        top_correlations = [
            {"feature": feature, "correlation": float(corr_matrix.at[feature, 'outcome'])}
            for feature in top_corr.index
        ]
    else:
        top_correlations = []

    # 4. Outlier detection (IQR method)
    q75 = df.quantile(0.75)
    q25 = df.quantile(0.25)
    iqr = q75 - q25
    mask = (df.select_dtypes(include="number") - df.select_dtypes(include="number").median()).abs() > (1.5 * iqr)
    outliers = df[mask.any(axis=1)]
    anomalies = outliers.to_dict(orient="records")

    # Statistical summary
    statistical_notes = (
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
        f"Descriptive stats indicate strongest linear relationships with 'outcome' are: "
        + ", ".join([c['feature'] for c in top_correlations])
        + "."
    )

    return {
        "plot_paths": plot_paths,
        "top_correlations": top_correlations,
        "anomalies": anomalies,
        "statistical_notes": statistical_notes
    }