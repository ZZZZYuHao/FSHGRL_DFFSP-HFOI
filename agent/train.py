"""Deprecated entry point kept for backwards compatibility: forwards to train.py at the repository root.

    python agent/train.py ...   ==   python train.py ...
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train import *  # noqa: F401,F403  (build_model_paras, set_seed, ... for old imports)
from train import main  # noqa: E402

if __name__ == "__main__":
    print("[deprecated] 'python agent/train.py' forwards to 'python train.py'", file=sys.stderr)
    main()
