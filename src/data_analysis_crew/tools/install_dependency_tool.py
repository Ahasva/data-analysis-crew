# src/data_analysis_crew/tools/install_dependency_tool.py
import subprocess
from crewai.tools import tool
from data_analysis_crew.utils.instructions import AVAILABLE_LIBRARIES

@tool("install_dependency")
def install_dependency(package: str) -> str:
    """
    Safely installs a Python package if it's in the approved list.
    """
    if package not in AVAILABLE_LIBRARIES:
        return f"❌ Package '{package}' is not in the approved list. Allowed: {', '.join(AVAILABLE_LIBRARIES)}"
    
    try:
        subprocess.check_call(["pip", "install", package])
        return f"✅ Installed package '{package}'."
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to install package '{package}': {e}"