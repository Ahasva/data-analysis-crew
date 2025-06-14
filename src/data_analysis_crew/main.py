#!/usr/bin/env python
"""
Entry-point for the Data-Analysis Crew pipeline
Run with  `crewai run`  or  `python -m data_analysis_crew.main`
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime, timezone
from data_analysis_crew.settings import (
    FILE_NAME, ROOT_FOLDER, DASHBOARD_FILE, OUTPUT_DIR, PLOT_PATH,
    REQUEST, AVAILABLE_MODELS, METRICS_BY_TYPE, EXPECTED_PLOTS
)
from data_analysis_crew.utils.instructions import INSTALL_LIB_TEMPLATE, AVAILABLE_LIBRARIES
from data_analysis_crew.crew import DataAnalysisCrew

def run() -> None:
    """Run the full data-analysis crew pipeline."""
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "plots").mkdir(parents=True, exist_ok=True)

    cleaned_filename = FILE_NAME.replace(".csv", "_cleaned.csv")

    inputs = {
        "dashboard_file": DASHBOARD_FILE,
        "root_folder": ROOT_FOLDER,
        "dataset_path": os.path.join(ROOT_FOLDER, FILE_NAME),
        "raw_path": os.path.join(ROOT_FOLDER, FILE_NAME),
        "cleaned_path": os.path.join(ROOT_FOLDER, cleaned_filename),
        "output_dir": OUTPUT_DIR,
        "plot_path": PLOT_PATH,
        "request": REQUEST,
        "install_hint": INSTALL_LIB_TEMPLATE,
        "available_libraries": AVAILABLE_LIBRARIES,
        "datetime": datetime.now(timezone.utc).isoformat(),
        "available_models": AVAILABLE_MODELS,
        "classification_models": AVAILABLE_MODELS["classification"],
        "regression_models": AVAILABLE_MODELS["regression"],
        "classification_metrics": METRICS_BY_TYPE["classification"],
        "regression_metrics": METRICS_BY_TYPE["regression"],
        "expected_plots": EXPECTED_PLOTS
    }

    print("🚧 DEBUG: pipeline inputs →")
    print(json.dumps(inputs, indent=2))

    try:
        crew = DataAnalysisCrew().crew()
        result = crew.kickoff(inputs=inputs)
        print("🚀 Dashboard launch has been delegated to the crew.")
        return result

    except Exception as e:
        raise RuntimeError(f"[RUN ERROR] Crew execution failed: {e}") from e

def train():
    """
    Train the crew with synthetic prompt and iterations.
    """
    inputs = {
        "data_source": "https://example.com/data.csv",
        "dataset_path": "https://example.com/data.csv",
        "request": "What are the main sales trends over the last 6 months?",
        "output_dir": "output",
    }

    try:
        DataAnalysisCrew().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise RuntimeError(f"[TRAIN ERROR] Failed to train the crew: {e}") from e

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        DataAnalysisCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise RuntimeError(f"[REPLAY ERROR] Failed to replay the task: {e}") from e

def test():
    """
    Test the crew execution and return results.
    """
    inputs = {
        "data_source": "https://example.com/data.csv",
        "dataset_path": "https://example.com/data.csv",
        "request": "What are the main sales trends over the last 6 months?",
        "output_dir": "output"
    }

    try:
        DataAnalysisCrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise RuntimeError(f"[TEST ERROR] Failed to test the crew: {e}") from e

def validate_final_summary(output_path: Path):
    """Simple checklist validator for final summary report."""
    summary_file = output_path / "final-insight-summary.md"
    if not summary_file.exists():
        print("❌ final-insight-summary.md not found.")
        return

    content = summary_file.read_text(encoding="utf-8").lower()

    checks = {
        "✔ summary section"      : "executive summary" in content,
        "✔ bullet insights"      : "-" in content or "*" in content,
        "✔ embedded visual"      : "<img " in content,
        "✔ metrics present"      : any(metric in content for metric in METRICS_BY_TYPE["classification"] + METRICS_BY_TYPE["regression"]),
        "✔ dashboard mentioned"  : "dashboard" in content or "http://localhost" in content
    }

    print("\n🔍 Final Report Checklist:")
    log_path = output_path / "final-checklist.log"
    with log_path.open("w", encoding="utf-8") as f:
        for desc, passed in checks.items():
            result = '✅' if passed else '❌'
            print(f"{desc:<30} {result}")
            f.write(f"{desc:<28} {'OK' if passed else 'MISSING'}\n")
    print(f"📝 Checklist log saved to: {log_path}")

# ─────────────────────────────── script launch ───────────────────────────────
if __name__ == "__main__":
    run()
