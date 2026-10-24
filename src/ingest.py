"""Download the raw goodbooks-10k CSV files.

Usage:
    python src/ingest.py           # skip files that already exist
    python src/ingest.py --force   # download everything again

Output directory is taken from the OUTPUT_PATH env variable (default: data/raw).
"""

import argparse
import logging
import os
from pathlib import Path

import requests

BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"
FILES = ["books.csv", "ratings.csv", "to_read.csv", "tags.csv", "book_tags.csv"]
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "data/raw"))

logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path) -> None:
    """Stream a file to disk; write to a temp file first so a failed download leaves no partial file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.replace(dest)


def main(force: bool = False) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = OUTPUT_PATH / name
        if dest.exists() and not force:
            logger.info("skip %s (already exists)", name)
            continue
        logger.info("downloading %s", name)
        download_file(f"{BASE_URL}/{name}", dest)
        logger.info("saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    main(force=parser.parse_args().force)
