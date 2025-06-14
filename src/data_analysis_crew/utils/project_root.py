from pathlib import Path

def get_project_root() -> Path:
    """Returns the root directory of the project (3 levels up from this file)."""
    return Path(__file__).resolve().parents[3]

def resolve_path(relative_path: str) -> Path:
    """Resolve a path string relative to the project root."""
    return get_project_root() / Path(relative_path)
