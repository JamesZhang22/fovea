import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resource_path(relative: str) -> Path:
    """Bundled resource path, PyInstaller extraction dir when frozen, repo root in dev."""
    base = Path(getattr(sys, "_MEIPASS", REPO_ROOT))
    return base / relative
