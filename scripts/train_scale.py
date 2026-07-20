#!/usr/bin/env python3
"""CLI for the config-driven scale trainer (see marc/train/trainer.py --help)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marc.train.trainer import main

if __name__ == "__main__":
    main()
