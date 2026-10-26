"""
backend/training/train_step3.py
================================
Thin wrapper around backend.training.train_luna16_3d --profile cpu.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.training.train_luna16_3d import run_cpu_training_pipeline

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    luna_root = Path(os.environ.get("LUNA16_ROOT", "data/LUNA16"))
    run_cpu_training_pipeline(luna_root=luna_root, seed=seed)
