# Deník projektu – Book Recommender

## Repo a venv
- Repo: `D:\03_Coding\project_repo\book_recommender` → GitHub `martincho48/book_recommender`
- Venv: `C:\Users\Martin\uv_venv\uv_book_recommender` (uv, Python 3.10) – mimo repo

## Záznamy

### 2026-10-24
- Repo založeno: `.gitignore`, `README.md` (plán projektu), tento deník.
- **Data:** interakce z goodbooks-10k (Goodreads API zavřené, Open Library nemá ID uživatelů); Open Library jen na obálky.
- **Plán:** 1) lokálně: ingest → ETL → baseline → two-tower → FastAPI → Docker; 2) GCP: GCS + BigQuery + Cloud Run, GitHub Actions (CI, CD, pipeline); 3) volitelně měsíční aktualizace z Open Library.
- `src/ingest.py` – stáhne 5 CSV do `data/raw/`.
- `notebooks/eda.ipynb` – EDA: 6M hodnocení, 53k uživatelů, 10k knih, data čistá.
- Používáme jen `ratings` + `books`; tagy a `to_read` zatím ne.
- `src/etl.py` – `books.parquet` + `interactions.parquet` (oddělené, propojené přes `book_id`), opraveno ISBN.
- `src/enrich.py` – obálky z Open Library (`cover_url`), cache `ol_covers.parquet`. Výsledek: Open Library 9 108, Goodreads 746, bez obálky 146. První běh ~28 min, další z cache ~45 s.
- Postup pipeline popsán v `docs/pipeline.md`; README má sekce Data sources a ETL (krok za krokem) + quick start.
- Nápady na později: souhrnná vydání (103 knih), hodnocení ≥ 4 jako pozitivní signál, korekce popularity.

- Commit `ddb7898` do `main` (data pipeline + EDA + dokumentace), zatím nepushnuto.

**Další krok:** push → baseline model.
