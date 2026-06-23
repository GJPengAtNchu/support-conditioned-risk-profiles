"""Placeholder data downloader for the public release.

Large files are intentionally not bundled with GitHub. After the Google Drive
archive is published, update GOOGLE_DRIVE_URL and extend this script if direct
programmatic download is desired.
"""

from __future__ import annotations

from pathlib import Path


GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1degj8NKU1FJTib9hocOrUMixivfM_5vt?usp=sharing"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("data/raw", "data/cache", "outputs/results", "outputs/figures", "outputs/tables"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    print("Created data/output directories.")
    print(f"Download large data and cached results from: {GOOGLE_DRIVE_URL}")


if __name__ == "__main__":
    main()
