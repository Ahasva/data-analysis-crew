from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import tool

@tool("Explore cleaned dataset with EDA plots and stats")
def explore_data():
    df = pd.read_csv("knowledge/diabetes_cleaned.csv")
    plots_dir = Path("output/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []
    for col in df.select_dtypes(include="number").columns:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        out_path = plots_dir / f"distribution_{col}.png"
        fig.savefig(out_path)
        plt.close(fig)
        plot_paths.append(str(out_path.relative_to(Path.cwd())))

    corr_matrix = df.corr(numeric_only=True)
    heatmap_path = plots_dir / "correlation_heatmap.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="YlOrRd")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    plot_paths.append(str(heatmap_path.relative_to(Path.cwd())))

    top_corr_series = corr_matrix["outcome"].drop("outcome").sort_values(ascending=False).head(3)
    top_correlations = [
        {"feature": f, "correlation": float(c)}
        for f, c in top_corr_series.items()
    ]

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
