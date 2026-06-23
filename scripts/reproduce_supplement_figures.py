"""Supplementary reproduction launcher.

This release keeps supplementary analyses as separate scripts. The commands
below are intentionally conservative and use fast/default modes where present.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    commands = [
        ["experiments/exp7_conflict_v2.py", "--outdir", "outputs"],
        ["scripts/postprocess_exp6_sensitivity.py", "--outdir", "outputs"],
        ["scripts/postprocess_exp7_conflict_v2.py", "--outdir", "outputs"],
    ]
    for command in commands:
        subprocess.run([sys.executable, str(root / command[0]), *command[1:]], cwd=root, check=True)


if __name__ == "__main__":
    main()
