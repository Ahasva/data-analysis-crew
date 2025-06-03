#!/usr/bin/env python
import os
import sys
import warnings
from data_analysis_crew.crew import DataAnalysisCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# FILE
DATA_FOLDER = "data"
FILE_NAME = "diabetes.csv"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, DATA_FOLDER, FILE_NAME)

# PROMPT
REQUEST="""
What are the main factors for Diabetes? 
Which feature in the given data has the gravest impact on the patient, 
resulting in diabetes?
"""


def run():
    """
    Run the crew.
    """
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    inputs = {
        "data_source": DATA_PATH,
        "request": REQUEST,
    }

    try:
        crew = DataAnalysisCrew().crew()
        print("\n🧭 Final Planned Task Order:")
        for task in crew.tasks:
            print(f"→ {task.agent.role}: {task.description[:50]}...")

        crew.kickoff(inputs=inputs)
    except Exception as e:
        raise RuntimeError(f"[RUN ERROR] Failed to run the crew: {e}") from e


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "data_source": "https://example.com/data.csv",
        "request": "What are the main sales trends over the last 6 months?",
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
        "request": "What are the main sales trends over the last 6 months?",
    }

    try:
        DataAnalysisCrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise RuntimeError(f"[TEST ERROR] Failed to test the crew: {e}") from e


if __name__ == "__main__":
    run()
