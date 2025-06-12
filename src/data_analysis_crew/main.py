#!/usr/bin/env python
"""
Entry-point for the Data-Analysis Crew pipeline
Run with  `crewai run`  or  `python -m data_analysis_crew.main`
"""
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone
from data_analysis_crew.utils.instructions import INSTALL_LIB_TEMPLATE, AVAILABLE_LIBRARIES
from data_analysis_crew.crew import DataAnalysisCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ───────────────────────────────────────── paths ─────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DATA_FILE     = PROJECT_ROOT / "knowledge" / "diabetes.csv"
REL_PATH_DATA = DATA_FILE.relative_to(PROJECT_ROOT)

OUTPUT_DIR_ABS = PROJECT_ROOT / "output"
PLOT_PATH_ABS  = OUTPUT_DIR_ABS / "plots"

# These are the relative paths passed into the crew
OUTPUT_DIR_REL = OUTPUT_DIR_ABS.relative_to(PROJECT_ROOT)
PLOT_PATH_REL  = PLOT_PATH_ABS.relative_to(PROJECT_ROOT)

# Create necessary folders
OUTPUT_DIR_ABS.mkdir(parents=True, exist_ok=True)
PLOT_PATH_ABS.mkdir(parents=True, exist_ok=True)

# Ensure dashboard.py exists
DASHBOARD_PY = PROJECT_ROOT / "dashboard.py"
if not DASHBOARD_PY.exists():
    raise FileNotFoundError(f"dashboard.py expected at {DASHBOARD_PY}")

# ───────────────────────────────────────── prompt ────────────────────────────────────────
REQUEST = """
What are the main factors for Diabetes?
Which feature in the given data has the gravest impact on the patient,
resulting in diabetes?
"""

# ─────────────────────────────────────── main helpers ─────────────────────────────────────
def run() -> None:
    """Run the full data-analysis crew pipeline."""
    inputs = {
        "dataset_path"       : str(REL_PATH_DATA),
        "request"            : REQUEST,
        "output_dir"         : str(OUTPUT_DIR_REL),                  # output
        "plot_path"          : str(PLOT_PATH_REL),                   # output/plots
        "install_hint"       : INSTALL_LIB_TEMPLATE,
        "available_libraries": AVAILABLE_LIBRARIES,
        "datetime"           : datetime.now(timezone.utc).isoformat()
    }

    try:
        crew = DataAnalysisCrew().crew()

        print("\n🧭 Planned task order:")
        for t in crew.tasks:
            print(f"→ {t.agent.role:<22} : {t.description.splitlines()[0]}")

        print("🧾 Passing dataset:", inputs["dataset_path"])
        print("📁 Output path    :", inputs["output_dir"])
        print("📂 Plot path      :", inputs["plot_path"])

        result = crew.kickoff(inputs=inputs)

        print("\n✅ Analysis completed.")
        print("🌐  Opening dashboard...")

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

# ─────────────────────────────── script launch ───────────────────────────────
if __name__ == "__main__":
    run()
    print("Dashboard launched 🚀  "
          "(Ctrl-C here won't stop it; close the browser tab or "
          "press Ctrl-C in the Streamlit terminal)")