import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import tool
from data_analysis_crew.utils.utils import to_posix_relative_path
from data_analysis_crew.utils.project_root import resolve_path, get_project_root
from data_analysis_crew.schemas import ExplorationOutput, FeatureCorrelation


@tool("explore_data")
def explore_data(cleaned_path: str, plot_path: str) -> ExplorationOutput:
    """
    Perform exploratory data analysis (EDA) on a cleaned dataset.
    - Generates histograms for all numeric features.
    - Creates a correlation heatmap.
    - Extracts top correlations with target.
    - Identifies outliers using IQR method.
    """

    cleaned_full_path = resolve_path(cleaned_path)
    plot_dir = resolve_path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cleaned_full_path)
    plot_paths = []

    # 1. Histograms
    for col in df.select_dtypes(include="number").columns:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        out_file = plot_dir / f"distribution_{col}.png"
        fig.savefig(out_file)
        plt.close(fig)
        plot_paths.append(to_posix_relative_path(out_file, get_project_root()))

    # 2. Correlation heatmap
    corr_matrix = df.corr(numeric_only=True)
    heatmap_file = plot_dir / "correlation_heatmap.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="YlOrRd")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_file)
    plt.close()
    plot_paths.append(to_posix_relative_path(heatmap_file, get_project_root()))

    # 3. Top correlations
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
        print("⚠️ 'outcome' column not found.")
        top_correlations = []

    # 4. Outliers (IQR)
    q75 = df.quantile(0.75)
    q25 = df.quantile(0.25)
    iqr = q75 - q25
    mask = (df.select_dtypes(include="number") - df.select_dtypes(include="number").median()).abs() > (1.5 * iqr)
    outliers = df[mask.any(axis=1)]
    anomalies = outliers.to_dict(orient="records")

    # 5. Summary
    statistical_notes = (
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
        f"Descriptive stats indicate strongest linear relationships with 'outcome' are: "
        + (", ".join([c['feature'] for c in top_correlations]) if top_correlations else "none found")
        + "."
    )

    # 6. Logging
    print(f"📊 Top correlated: {[f['feature'] for f in top_correlations]}")
    print(f"🖼️  Saved {len(plot_paths)} visualizations to: {plot_path}")

    # 7. Output using schema
    return ExplorationOutput(
        plot_paths=plot_paths,
        top_correlations=[FeatureCorrelation(**c) for c in top_correlations],
        anomalies=anomalies or [],
        statistical_notes=statistical_notes
    )
