from pathlib import Path

def to_posix_relative_path(path: Path, root: Path) -> str:
    """
    Return a clean POSIX-style relative path (with forward slashes),
    relative to root directory.
    """
    rel_path = path.relative_to(root)
    return rel_path.as_posix()
