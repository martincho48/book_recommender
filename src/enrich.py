"""Add Open Library cover URLs to the processed books.

Usage:
    python src/enrich.py

Looks up the Open Library cover ID (cover_i) for every book with an ISBN, in batches
via search.json. Found IDs are cached in ol_covers.parquet, so later runs only ask
for books that do not have a cover yet. Adds two columns to books.parquet:
    ol_cover_id – Open Library cover ID (NaN if not found)
    cover_url   – Open Library cover, falls back to the Goodreads image_url

Paths: DATA_PATH (default: data/processed).
Set OPENLIBRARY_CONTACT (e.g. an email) to identify the app – Open Library then allows 3 req/s instead of 1.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_PATH = Path(os.environ.get("DATA_PATH", "data/processed"))
BOOKS_FILE = DATA_PATH / "books.parquet"
CACHE_FILE = DATA_PATH / "ol_covers.parquet"

SEARCH_URL = "https://openlibrary.org/search.json"
COVER_URL = "https://covers.openlibrary.org/b/id/{}-M.jpg"
BATCH_SIZE = 50

CONTACT = os.environ.get("OPENLIBRARY_CONTACT")
HEADERS = {"User-Agent": f"book_recommender ({CONTACT})" if CONTACT else "book_recommender"}
REQUEST_DELAY = 0.4 if CONTACT else 1.0  # seconds, keeps us under the Open Library rate limit

logger = logging.getLogger(__name__)


def fetch_cover_ids(isbns: list[str], retries: int = 3) -> dict[str, int]:
    """Return {isbn: cover_i} for one batch of ISBNs found on Open Library."""
    params = {"q": f"isbn:({' OR '.join(isbns)})", "fields": "cover_i,isbn", "limit": 200}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == retries:
                raise
            logger.warning("request failed (%s), retry %d/%d", e, attempt, retries)
            time.sleep(5 * attempt)

    wanted = set(isbns)
    found = {}
    for doc in resp.json()["docs"]:
        if "cover_i" not in doc:
            continue
        for isbn in wanted.intersection(doc.get("isbn", [])):
            found.setdefault(isbn, doc["cover_i"])
    return found


def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)
    return pd.DataFrame({"isbn": pd.Series(dtype="str"), "ol_cover_id": pd.Series(dtype="int64")})


def main() -> None:
    books = pd.read_parquet(BOOKS_FILE).drop(columns=["ol_cover_id", "cover_url"], errors="ignore")
    cache = load_cache()

    todo = sorted(set(books["isbn"].dropna()) - set(cache["isbn"]))
    logger.info("cached covers: %d, ISBNs to look up: %d", len(cache), len(todo))

    n_batches = -(-len(todo) // BATCH_SIZE)
    for i, start in enumerate(range(0, len(todo), BATCH_SIZE), 1):
        batch = todo[start:start + BATCH_SIZE]
        t0 = time.monotonic()
        found = fetch_cover_ids(batch)
        if found:
            # save after every batch, so an interrupted run keeps what it already found
            new = pd.DataFrame({"isbn": list(found), "ol_cover_id": list(found.values())})
            cache = pd.concat([cache, new], ignore_index=True)
            cache.to_parquet(CACHE_FILE, index=False)
        logger.info("batch %d/%d: %d/%d found (%.1fs)", i, n_batches, len(found), len(batch), time.monotonic() - t0)
        time.sleep(REQUEST_DELAY)

    books = books.merge(cache, on="isbn", how="left")
    books["ol_cover_id"] = books["ol_cover_id"].astype("Int64")
    ol_url = books["ol_cover_id"].map(lambda x: COVER_URL.format(int(x)), na_action="ignore")
    books["cover_url"] = ol_url.fillna(books["image_url"])
    books.to_parquet(BOOKS_FILE, index=False)

    n = len(books)
    n_ol = books["ol_cover_id"].notna().sum()
    n_any = books["cover_url"].notna().sum()
    logger.info("covers: Open Library %d, Goodreads fallback %d, none %d (of %d books)",
                n_ol, n_any - n_ol, n - n_any, n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
