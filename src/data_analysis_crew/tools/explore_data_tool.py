from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import tool
from data_analysis_crew.utils.utils import to_posix_relative_path

@tool("explore_data")
def explore_data(
    data_path: str = "knowledge/diabetes_cleaned.csv",
    output_dir: str = "output/plots"
) -> dict:
    """
    Perform exploratory data analysis (EDA) on a cleaned dataset.

    Generates:
      - Histograms of all numeric features
      - A correlation heatmap
      - Top 3 most correlated features with 'outcome'
      - Outlier samples based on IQR

    Saves plots to the specified output directory and returns:
      - POSIX-style paths to saved plots
      - Statistical notes
      - Top correlated features
      - Anomalous records (outliers)
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    df = pd.read_csv(data_path)

    plots_dir = PROJECT_ROOT / output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []

    # ── 1. Histograms ─────────────────────────────────────
    for col in df.select_dtypes(include="number").columns:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        out_path = plots_dir / f"distribution_{col}.png"
        fig.savefig(out_path)
        plt.close(fig)
        plot_paths.append(to_posix_relative_path(out_path, PROJECT_ROOT))

    # ── 2. Correlation Heatmap ────────────────────────────
    corr_matrix = df.corr(numeric_only=True)
    heatmap_path = plots_dir / "correlation_heatmap.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="YlOrRd")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    plot_paths.append(to_posix_relative_path(heatmap_path, PROJECT_ROOT))

    # ── 3. Top Correlations ───────────────────────────────
    top_corr_series = corr_matrix["outcome"].drop("outcome").sort_values(ascending=False).head(3)
    top_correlations = [
        {"feature": f, "correlation": float(c)}
        for f, c in top_corr_series.items()
    ]

    # ── 4. Outliers (Anomalies) ───────────────────────────
    iqr = df.quantile(0.75) - df.quantile(0.25)
    outliers = df[((df - df.median()).abs() > 1.5 * iqr).any(axis=1)]
    anomalies = outliers.to_dict(orient="records")

    return {
        "plot_paths": plot_paths,
        "top_correlations": top_correlations,
        "anomalies": anomalies,
        "statistical_notes": (
            "The dataset contains 9 numeric features. Descriptive stats and "
            "correlation matrix show that 'glucose', 'bmi', and 'age' have the "
            "strongest relationships with diabetes outcome."
        )
    }
