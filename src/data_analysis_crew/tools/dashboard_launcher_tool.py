# ── dashboard_launcher_tool.py ──
import socket
import subprocess
import webbrowser
import time
from crewai.tools import tool
from data_analysis_crew.utils.project_root import resolve_path

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
    script_path = resolve_path(path)
    subprocess.Popen(
        ["streamlit", "run", str(script_path), "--server.port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    
    def wait_for_port(port, timeout=15):
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    return
            except OSError:
                time.sleep(0.5)

    wait_for_port(port)

    url = f"http://localhost:{port}"
    try:
        webbrowser.open_new(url)
    except Exception:
        pass  # Fail silently in headless or remote environments

    print(f"✅ Dashboard launched → {url}")
    return f"Dashboard launched on {url}"