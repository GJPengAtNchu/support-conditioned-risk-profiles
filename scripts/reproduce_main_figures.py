"""Convenience launcher for regenerating main experiment outputs.

This script intentionally delegates to the public experiment entry points.
Use `--mode cached` after downloading cached CSV/JSON files from Google Drive;
use `--mode full` to rerun experiments. The full mode can be slow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FAST_COMMANDS = [
    ["experiments/exp2_support_gap.py", "--fast"],
    ["experiments/exp3_local_polynomial.py", "--fast"],
    ["experiments/exp4_krr_decomposition.py", "--fast"],
    ["experiments/exp5_support_only_insufficiency.py", "--fast"],
    ["experiments/exp6_model_selection.py", "--fast"],
    ["experiments/exp7_acquisition.py", "--fast"],
    ["experiments/exp8_real_data.py", "--fast"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cached", "full"), default="cached")
    parser.add_argument("--outdir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.mode == "cached":
        print("Cached mode expects downloaded results under outputs/.")
        print("Use the individual plotting/experiment scripts if a figure needs regeneration.")
        return
    for command in FAST_COMMANDS:
        subprocess.run(
            [sys.executable, str(root / command[0]), *command[1:], "--outdir", args.outdir],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
