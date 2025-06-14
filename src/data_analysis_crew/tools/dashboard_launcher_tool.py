# ── dashboard_launcher_tool.py ──
from pathlib import Path
import subprocess
import webbrowser
import time
from crewai.tools import tool

@tool("launch_dashboard")
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
    subprocess.Popen([
        "streamlit", "run", str(script_path),
        "--server.port", str(port)
    ])

    # 🔄 Optional: wait a few seconds for Streamlit server to start
    time.sleep(15)

    url = f"http://localhost:{port}"
    try:
        webbrowser.open_new(url)
    except Exception:
        pass  # Fail silently in headless or remote environments

    print(f"✅ Dashboard launched → {url}")
    return f"Dashboard launched on {url}"