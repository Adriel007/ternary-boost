"""Test configuration — ensures workspace packages are importable."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent

for _pkg in ("shared", "pt_bitnet", "paretoq", "tequila", "eval"):
    _src = _project_root / _pkg / "src"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
