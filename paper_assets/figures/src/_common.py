"""Shared constants of the figure scripts: paths, method labels, skip helper."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# FSHGRL_RESULT_DIR / FSHGRL_FIG_DIR redirect the inputs and outputs (used by scripts/run_00_smoke.py)
RESULT = Path(os.environ.get("FSHGRL_RESULT_DIR", ROOT / "result"))
OUT_DIR = Path(os.environ.get("FSHGRL_FIG_DIR", ROOT / "paper_assets" / "figures" / "generated"))


def require(*paths: Path) -> bool:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print("[SKIP] missing input(s): " + ", ".join(str(p) for p in missing))
        return False
    return True
