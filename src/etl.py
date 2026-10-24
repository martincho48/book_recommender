"""Clean raw goodbooks-10k data and save it as parquet.

Usage:
    python src/etl.py

Reads books.csv and ratings.csv from INPUT_PATH (default: data/raw) and writes
books.parquet and interactions.parquet to OUTPUT_PATH (default: data/processed).
"""

import logging
import os
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "data/raw"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "data/processed"))

BOOK_COLUMNS = [
    "book_id", "goodreads_book_id", "title", "original_title", "authors",
    "original_publication_year", "language_code", "isbn", "isbn13", "image_url",
]
ENGLISH_CODES = {"eng", "en-US", "en-GB", "en-CA", "en"}

logger = logging.getLogger(__name__)


def isbn10_to_isbn13(isbn10: str) -> str:
    """Prefix 978 to the first 9 digits and compute the ISBN-13 check digit."""
    core = "978" + isbn10[:9]
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(core))
    return core + str((10 - total % 10) % 10)


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """Keep useful columns and fix formats. Rows with NaN are kept – every book has ratings."""
    df = books[BOOK_COLUMNS].copy()

    # isbn lost its leading zeros in the source file
    df["isbn"] = df["isbn"].str.zfill(10)
    # isbn13 in the source has only 12 significant digits (check digit lost) -> rebuild it from isbn
    df["isbn13"] = df["isbn"].map(isbn10_to_isbn13, na_action="ignore")

    df["original_title"] = df["original_title"].fillna(df["title"])
    df["original_publication_year"] = df["original_publication_year"].astype("Int64")
    df["language_code"] = df["language_code"].replace(dict.fromkeys(ENGLISH_CODES, "en"))
    # placeholder cover -> NaN, so the app can fall back to another source later
    df["image_url"] = df["image_url"].mask(df["image_url"].str.contains("nophoto"))

    return df


def clean_ratings(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows: NaN, duplicate (user, book) pairs, ratings outside 1–5, unknown books."""
    n = len(ratings)
    df = ratings.dropna()
    df = df.drop_duplicates(["user_id", "book_id"], keep="last")
    df = df[df["rating"].between(1, 5)]
    df = df[df["book_id"].isin(books["book_id"])]
    df = df.astype({"user_id": "int32", "book_id": "int32", "rating": "int8"})
    logger.info("ratings: removed %d of %d rows", n - len(df), n)
    return df.reset_index(drop=True)


def validate(books: pd.DataFrame, interactions: pd.DataFrame) -> None:
    assert books["book_id"].is_unique, "duplicate book_id in books"
    assert interactions[["user_id", "book_id", "rating"]].notna().all().all(), "NaN in interactions"
    assert not interactions.duplicated(["user_id", "book_id"]).any(), "duplicate (user, book) pairs"
    assert interactions["rating"].between(1, 5).all(), "rating outside 1–5"
    assert interactions["book_id"].isin(books["book_id"]).all(), "interaction with unknown book"
    assert books["book_id"].isin(interactions["book_id"]).all(), "book without any interaction"


def main() -> None:
    books = clean_books(pd.read_csv(INPUT_PATH / "books.csv"))
    interactions = clean_ratings(pd.read_csv(INPUT_PATH / "ratings.csv"), books)
    validate(books, interactions)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    books.to_parquet(OUTPUT_PATH / "books.parquet", index=False)
    interactions.to_parquet(OUTPUT_PATH / "interactions.parquet", index=False)
    logger.info("saved %d books and %d interactions to %s", len(books), len(interactions), OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
