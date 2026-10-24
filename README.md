# Book Recommender

Enter a book, get similar books — based on what other users read and rate.

> 🚧 Work in progress

## How it works

1. **Pipeline** – download goodbooks-10k, clean it, add covers, compute similar books (item-item)
2. **Research** – compare popularity, item-item, ALS and a two-tower neural network; item-item wins for "similar books"
3. **Serving** – app on GCP that scales to zero, so it costs money only when a request comes in

## Project structure

```
prod/pipeline.py            production: raw data -> books, interactions, similar books
prod/app.py                 production: Streamlit app – pick a book or an author, see similar books with genres and descriptions
research/eda.ipynb          data exploration
research/experiments.ipynb  model comparison: split, features, models, training, evaluation, Excel report
prod/test_pipeline.py       tests of the production code
data/                       raw/, cache/ (covers), processed/ – created by the pipeline, not in git
research/output/            results.xlsx, data_*.parquet (split, features), model_*.pt – not in git
```

## Data sources

| Source | What we take | Why |
|---|---|---|
| [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) | 6M ratings from 53k users on 10k books, book metadata | The model needs user–book interactions. Goodreads closed its API in 2020; this is a public snapshot of it. |
| [Open Library](https://openlibrary.org/developers/api) | Book covers | Larger and more complete covers than goodbooks-10k. Open Library has no per-user ratings, so it is used only to enrich books. |
| [Wikipedia](https://en.wikipedia.org/api/rest_v1/) | Short author info (description + first paragraph) and author photo (URL only) | Real facts instead of generated text; found for 2,876 of 3,888 authors. Text under CC BY-SA, the app links the source. |

goodbooks-10k by [zygmuntz](https://github.com/zygmuntz) ([FastML](http://fastml.com/goodbooks-10k)) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Used: `ratings.csv`, `books.csv`, reader tags (`book_tags.csv`, `tags.csv`) for genres and `books_xml.zip`
(Goodreads XML files from the same repository) for book descriptions.

## Pipeline – `prod/pipeline.py`

**1. Ingest** – downloads the goodbooks-10k files into `data/raw/`. Files that already exist are skipped.

**2. ETL**
- *Books:* keeps the useful columns, fixes broken ISBNs, unifies language codes, splits `authors` into `author` (main author) and `contributors` (illustrators, translators, introductions). Books with missing values are kept – every book has ratings.
- *Ratings:* removes invalid rows (missing values, duplicates, ratings outside 1–5).
- *Validation:* checks that every rating belongs to a book and every book has ratings; stops if not.
- *Genres:* 1–4 of 30 genres per book from reader tags – see [How genres are built](#how-genres-are-built).
- *Descriptions:* from the Goodreads XML files (9,685 books).
- *Author info:* short description and bio from Wikipedia, kept only when the page describes a writer
  (otherwise it is likely a different person with the same name). Looked up once and cached in `data/cache/authors.parquet`.
- *Covers:* looks up covers on Open Library by ISBN (Goodreads cover as fallback). Results are cached in `data/cache/`,
  so later runs only ask for books still missing a cover. Result: 9,108 covers from Open Library, 746 from Goodreads, 146 without a cover.
- Output: `books.parquet` (10k books) and `interactions.parquet` (6M ratings) in `data/processed/`, linked by `book_id`.

**3. Similar books** – item-item similarity (see below) from all ratings ≥ 4 → the 20 most similar books for every book
→ `similar_books.parquet` (`book_id`, `rank`, `similar_book_id`, `score`; 10,000 × 20 rows).
The answers are computed once here, so the app only looks them up instead of crunching 6M ratings on every request.
The similarity uses all books, but the 76 books with titles in Arabic, Japanese, Greek or Cyrillic script (`title_is_latin`)
are neither listed nor recommended – the app is English.

```
interactions ──(item-item)──► similar_books
app: "The Hobbit" → books (find book_id) → similar_books (20 similar) → books (titles, authors, covers)
```

| Output in `data/processed/` | Content | Used by |
|---|---|---|
| `books.parquet` | 10k books – title, author, ISBN, cover | app, research |
| `interactions.parquet` | 6M ratings – who rated which book | step 3, research (not the app) |
| `similar_books.parquet` | 20 most similar books for every book | app |

### How genres are built

goodbooks-10k has no genre column – only 34,252 free-form tags that readers put on books (`book_tags.csv`, up to 100 tags per book with vote counts).
Most of them are not genres, so they are turned into genres in three steps (`GENRES`, `tag_genres`, `pick_genres` in `prod/pipeline.py`):

1. **Statistical extract.** Tags used on at least 150 books, minus shelf and status tags (`to-read`, `owned`, `kindle`, `favorites`,
   `read-in-2015`, `5-stars` …) → 548 candidates. The most frequent tags are almost all shelves – `fantasy` is only around the 37th place.
2. **30 genres with their variants.** From the candidates, 30 genres were picked by hand, each with the tags that mean it,
   e.g. `science-fiction` ← `sci-fi`, `scifi`, `sf`, `space-opera`. A tag can count for two genres (`sff` → fantasy and science fiction).
3. **Genres per book.** The votes of a genre's tags are summed per book. A book gets every genre with at least 30 % of the votes
   of its strongest genre, max 4 – so one reader tagging *Harry Potter* as `horror` does not make it a horror book.

Result: every book has 1–4 genres, e.g. *The Hobbit* → fantasy, classics; *The Hunger Games* → young adult, dystopian, science fiction.
Guessing genres from book descriptions with a local Hugging Face embedding model was also tried; it agreed with the reader tags
for only 33 % of books, so the tags are used.

## Research – `research/experiments.ipynb`

Compares four ways to find similar books. Details and explanations are in the notebook; per-user results,
the top 50 for sample users and similar books for popular titles are in `research/output/results.xlsx`.

| Model | How it finds similar books |
|---|---|
| Popularity | ignores the book – most liked books (baseline) |
| **Item-item** | shared readers: `shared / √(readers of A × readers of B)` – no training |
| ALS | matrix factorization (`implicit`) – cosine of learned book vectors |
| Two-tower | neural network (PyTorch) – learns user and book vectors from (user, liked book) pairs; v2 adds author, year, language, genres and a description vector |

**Data and split.** Ratings ≥ 4 = "liked". Each user's liked books are split at random 80 / 10 / 10 into train / validation / test
(no timestamps, so no time split). Validation is used for tuning, test only once at the end. Books the user already read are never recommended.

**Metrics.** *Book hit rate@10* is the main one – the app scenario: from one book the user liked, are any of the 10 most similar books
among their hidden test books? *Recall@K* / *NDCG@K* measure recommendations for a user, *coverage* how much of the catalogue gets recommended.

| Model | Book hit rate@10 | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|
| Popularity | 0.289 | 0.052 | 0.060 | 0.6 % |
| **Item-item** | **0.344** | 0.175 | 0.201 | 38 % |
| ALS | 0.285 | 0.137 | 0.140 | 31 % |
| Two-tower | 0.294 | 0.210 | 0.225 | 98 % |
| Two-tower + book features | 0.292 | **0.212** | **0.227** | **98 %** |

The two-tower model is best at recommending books to a user, but for book → similar books simple item-item is clearly better,
and neither ALS nor book features changed that. **The app uses item-item.**

## How to run

```bash
pip install -r requirements-dev.txt     # pipeline, research and tests (runs locally)
python prod/pipeline.py                 # ingest -> etl -> similar books (first run ~1.5 h: covers, author info, translations – all cached)
streamlit run prod/app.py --server.enableStaticServing true   # app on http://localhost:8501
pytest
jupyter notebook research/experiments.ipynb
```

`requirements.txt` holds only what the app needs (Streamlit, pandas, pyarrow) – that is what gets deployed.

## Deployment and cost safety (planned)

The app is a **Streamlit** app (`prod/app.py`), to be deployed on **Cloud Run**, reading only `books.parquet` and `similar_books.parquet`.
The data is static, so only the app runs on GCP – the pipeline runs locally and its two output files are shipped with the app.
Cloud Run scales to zero, but Streamlit keeps a connection open while the page is open and Cloud Run bills that time,
so the deployment gets several safety limits:

| Safety limit | Protects against | Reacts |
|---|---|---|
| `--max-instances=1` | traffic spikes, bots, attacks | immediately – never more than one instance |
| `--timeout=300` | endless connections | after 5 min |
| idle check in the page (JavaScript) | forgotten open tabs | after 2 min without activity the page closes the connection |
| budget 5 USD with e-mail alerts | – | at 50 / 90 / 100 % |
| kill switch: budget → Pub/Sub → function removes public access (`allUsers`) | slow cost leaks, e.g. a bot holding the page for days | when the budget is reached; nothing is deleted, one command restores it |

Worst case ≈ 5–7 USD per month (budget + a few hours of billing delay); normal portfolio traffic fits the Cloud Run free tier.

## Tech stack

- Google Cloud Platform
- Docker
- GitHub Actions (CI/CD and orchestration)
- Python

## Roadmap

- [x] Data ingestion (local)
- [x] ETL pipeline (local)
- [x] Model comparison: popularity, item-item, ALS, two-tower (with and without book features)
- [x] Similar books for the app (item-item)
- [x] Streamlit app (local)
- [ ] Deployment on Cloud Run with cost safety limits + GitHub Actions
