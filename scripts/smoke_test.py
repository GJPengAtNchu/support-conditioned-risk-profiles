"""Lightweight import and syntax smoke test for the release folder."""

from __future__ import annotations

import py_compile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py_files = sorted((root / "experiments").glob("*.py")) + sorted((root / "scripts").glob("*.py"))
    for path in py_files:
        py_compile.compile(str(path), doraise=True)
    print(f"Compiled {len(py_files)} Python files successfully.")


if __name__ == "__main__":
    main()
