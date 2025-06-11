"""
Tool for reporting agent for launching the Streamlit dashboard
"""
# ───────────────────────────── dashboard_launcher_tool.py ─────────────────────────────
from pathlib import Path
import subprocess
import webbrowser
import time
from crewai.tools import tool


@tool("Launch dashboard via Streamlit")
def launch_dashboard(path: str = "dashboard.py", port: int = 8501) -> str:
    """
    Start the Streamlit dashboard and open it in the browser.

    Parameters
    ----------
    path : str
        Relative or absolute path to the dashboard script (default: "dashboard.py").
    port : int
        Port on which Streamlit should run (default: 8501).

    Returns
    -------
    str
        Confirmation string with the URL that was opened.
    """
    script_path = Path(path).expanduser().resolve()
    subprocess.Popen(["streamlit", "run", str(script_path), "--server.port", str(port)])

    time.sleep(3)            # give Streamlit a moment to start
    url = f"http://localhost:{port}"
    webbrowser.open_new(url)
    print(f"Dashboard launched 🚀 → {url}")
    return f"Dashboard launched on {url}"
