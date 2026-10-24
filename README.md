# Book Recommender

Enter a book, get similar books — based on what other users read and rate.

> 🚧 Work in progress

## How it works

1. **Data ingestion** – download book and user rating data (goodbooks-10k)
2. **ETL on GCP** – clean and transform the data in the cloud
3. **Two-tower model** – learn book and user embeddings; recommend books close to the given one
4. **Serving** – app on GCP that scales to zero, so it costs money only when a request comes in

## Data sources

| Source | What we take | Why |
|---|---|---|
| [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) | 6M ratings from 53k users on 10k books, book metadata | The model needs user–book interactions. Goodreads closed its API in 2020; this is a public snapshot of it. |
| [Open Library](https://openlibrary.org/developers/api) | Book covers | Larger and more complete covers than goodbooks-10k. Open Library has no per-user ratings, so it is used only to enrich books. |

goodbooks-10k by [zygmuntz](https://github.com/zygmuntz) ([FastML](http://fastml.com/goodbooks-10k)) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Only `ratings.csv` and `books.csv` are used for now; tags and to-read lists are kept for later experiments.

## ETL

**1. Ingest** – `src/ingest.py`
Downloads the raw goodbooks-10k CSV files into `data/raw/`. Files that already exist are skipped.

**2. Clean** – `src/etl.py`
- *Books:* keeps the useful columns, fixes broken ISBNs, unifies language codes. Books with missing values are kept – every book has ratings.
- *Ratings:* removes invalid rows (missing values, duplicates, ratings outside 1–5).
- *Validation:* checks that every rating belongs to a book and every book has ratings; stops if not.
- Output: `books.parquet` (10k books) and `interactions.parquet` (6M ratings) in `data/processed/`. The tables stay separate, linked by `book_id`.

**3. Enrich** – `src/enrich.py`
Looks up covers on Open Library by ISBN and adds a `cover_url` column to `books.parquet` (Goodreads cover as fallback).
Results are cached, so later runs only ask for books that are still missing a cover.
Result: 9,108 covers from Open Library, 746 from Goodreads, 146 without a cover.

## Quick start

```bash
pip install -r requirements.txt
python src/ingest.py   # download raw CSVs
python src/etl.py      # clean -> books.parquet + interactions.parquet
python src/enrich.py   # add book covers
```

Details: [docs/pipeline.md](docs/pipeline.md)

## Tech stack

- Google Cloud Platform
- Docker
- GitHub Actions (CI/CD and orchestration)
- Python

## Roadmap

- [x] Data ingestion (local)
- [x] ETL pipeline (local)
- [ ] Two-tower model
- [ ] Deployment and orchestration
