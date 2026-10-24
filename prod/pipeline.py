"""Production data pipeline: raw goodbooks-10k -> clean data -> similar books for the app.

Usage:
    python prod/pipeline.py                 # all steps
    python prod/pipeline.py etl similar     # only some steps
    python prod/pipeline.py ingest --force  # download again

Steps:
    ingest   download 5 CSVs + books_xml.zip (book descriptions) into data/raw/
    etl      clean books and ratings, validate, add genres (reader tags), descriptions (Goodreads XML, non-English ones
             translated to English by a local Hugging Face model),
             author info (Wikipedia) and covers (Open Library) -> data/processed/books.parquet, interactions.parquet
             Translations, author info and covers are made once and cached in data/cache/.
    similar  item-item similarity (cosine of shared readers) -> data/processed/similar_books.parquet
             (only books with Latin-script titles are listed and recommended)

DATA_PATH (default: <repo>/data) is the root folder; on GCP it can point to Cloud Storage.
OPENLIBRARY_CONTACT (e.g. an email) identifies the app – Open Library then allows 3 req/s instead of 1.
"""

import argparse
import html
import logging
import os
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

DATA_PATH = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parents[1] / "data"))   # <repo>/data
RAW = DATA_PATH / "raw"
CACHE = DATA_PATH / "cache"
PROCESSED = DATA_PATH / "processed"

# --- ingest
SOURCE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"
SOURCE_FILES = ["books.csv", "ratings.csv", "to_read.csv", "tags.csv", "book_tags.csv", "books_xml/books_xml.zip"]

# --- etl
BOOK_COLUMNS = [
    "book_id", "goodreads_book_id", "title", "original_title", "authors",
    "original_publication_year", "language_code", "isbn", "isbn13", "image_url",
]
ENGLISH_CODES = {"eng", "en-US", "en-GB", "en-CA", "en"}

# --- genres: genre -> reader tags that mean it (a tag may count for more genres, e.g. sff).
# Picked from the most frequent tags after removing shelves like to-read / owned / kindle.
GENRES = {
    "fantasy": ["fantasy", "urban-fantasy", "high-fantasy", "epic-fantasy", "ya-fantasy", "magic",
                "science-fiction-fantasy", "fantasy-scifi", "sff"],
    "science-fiction": ["science-fiction", "sci-fi", "scifi", "sf", "space-opera", "science-fiction-fantasy",
                        "fantasy-scifi", "sff", "speculative-fiction"],
    "dystopian": ["dystopian", "dystopia", "post-apocalyptic"],
    "romance": ["romance", "contemporary-romance", "love", "chick-lit", "new-adult", "erotica", "paranormal-romance"],
    "paranormal": ["paranormal", "supernatural", "paranormal-romance", "vampires"],
    "mystery": ["mystery", "mysteries", "detective", "murder-mystery", "mystery-crime", "crime-mystery",
                "mystery-thriller", "mystery-suspense"],
    "thriller": ["thriller", "thrillers", "suspense", "suspense-thriller", "crime-thriller", "espionage", "spy",
                 "mystery-thriller"],
    "crime": ["crime", "crime-fiction", "murder", "true-crime"],
    "horror": ["horror"],
    "historical-fiction": ["historical-fiction", "historical"],
    "classics": ["classics", "classic"],
    "young-adult": ["young-adult", "ya", "teen", "ya-fiction", "young-adult-fiction", "ya-books", "ya-lit", "teen-fiction"],
    "children": ["children", "childrens", "kids", "middle-grade", "juvenile", "children-s-books", "childrens-books",
                 "kids-books", "children-s", "picture-books"],
    "humor": ["humor", "humour", "funny", "comedy"],
    "contemporary": ["contemporary", "contemporary-fiction", "realistic-fiction"],
    "literary-fiction": ["literary-fiction", "literary", "modern-fiction"],
    "adventure": ["adventure", "action", "action-adventure"],
    "non-fiction": ["non-fiction", "nonfiction"],
    "biography-memoir": ["biography", "memoir", "memoirs", "biography-memoir", "autobiography"],
    "history": ["history"],
    "philosophy": ["philosophy"],
    "religion": ["religion", "christian", "christianity", "spirituality", "christian-fiction"],
    "self-help": ["self-help", "psychology", "business", "economics"],
    "science": ["science", "nature"],
    "politics": ["politics", "feminism"],
    "war": ["war"],
    "poetry": ["poetry"],
    "graphic-novels": ["graphic-novels", "graphic-novel", "comics", "manga"],
    "short-stories": ["short-stories"],
    "mythology": ["mythology", "fairy-tales"],
}
GENRE_MIN_RELATIVE = 0.3   # a book gets a genre with >= 30 % of the votes of its strongest genre
MAX_GENRES = 4

# --- covers
SEARCH_URL = "https://openlibrary.org/search.json"
COVER_URL = "https://covers.openlibrary.org/b/id/{}-M.jpg"
COVER_BATCH = 50
CONTACT = os.environ.get("OPENLIBRARY_CONTACT")
HEADERS = {"User-Agent": f"book_recommender ({CONTACT})" if CONTACT else "book_recommender"}
REQUEST_DELAY = 0.4 if CONTACT else 1.0  # seconds, keeps us under the Open Library rate limit

# --- translation of non-English descriptions to English (local Hugging Face model, run once, cached)
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
# langdetect code -> NLLB code; descriptions in other detected languages are kept as they are
NLLB_LANGS = {
    "ar": "arb_Arab", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "id": "ind_Latn", "pt": "por_Latn",
    "ja": "jpn_Jpan", "it": "ita_Latn", "pl": "pol_Latn", "fa": "pes_Arab", "nl": "nld_Latn", "hr": "hrv_Latn",
    "da": "dan_Latn", "ru": "rus_Cyrl", "sv": "swe_Latn", "no": "nob_Latn", "fi": "fin_Latn", "tr": "tur_Latn",
    "el": "ell_Grek", "he": "heb_Hebr", "zh-cn": "zho_Hans", "zh-tw": "zho_Hant", "ko": "kor_Hang", "cs": "ces_Latn",
    "hu": "hun_Latn", "ro": "ron_Latn", "vi": "vie_Latn", "ca": "cat_Latn", "uk": "ukr_Cyrl", "bg": "bul_Cyrl",
}

# --- authors (Wikipedia, text under CC BY-SA – the app shows the source)
WIKI_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_HEADERS = {"User-Agent": "book_recommender/1.0 (https://github.com/martincho48/book_recommender)"}
WIKI_WORKERS = 2      # parallel requests; more gets "429 Too Many Requests"
# the Wikipedia page must describe a writer, otherwise it is probably a different person with the same name
WRITER_WORDS = re.compile(r"writer|author|novelist|poet|playwright|journalist|essayist|screenwriter|illustrator|"
                          r"cartoonist|historian|philosopher|biographer|columnist|scientist|theologian|memoirist|"
                          r"humorist|comedian|lyricist|dramatist", re.I)

# --- similar
MIN_RATING = 4        # rating >= 4 = "liked"
TOP_SIMILAR = 20

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- ingest

def download_file(url: str, dest: Path) -> None:
    """Stream a file to disk; write to a temp file first so a failed download leaves no partial file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.replace(dest)


def ingest(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        dest = RAW / Path(name).name
        if dest.exists() and not force:
            logger.info("skip %s (already exists)", dest.name)
            continue
        logger.info("downloading %s", name)
        download_file(f"{SOURCE_URL}/{name}", dest)
        logger.info("saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


# ---------------------------------------------------------------- etl

def isbn10_to_isbn13(isbn10: str) -> str:
    """Prefix 978 to the first 9 digits and compute the ISBN-13 check digit."""
    core = "978" + isbn10[:9]
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(core))
    return core + str((10 - total % 10) % 10)


def is_latin(text: str) -> bool:
    """True when every letter is from the Latin script (accents included); digits and punctuation don't count.
    Checks the script, not the language – reliable even for short texts like titles."""
    return all(unicodedata.name(ch, "").startswith("LATIN") for ch in text if ch.isalpha())


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """Keep useful columns and fix formats. Rows with NaN are kept – every book has ratings."""
    df = books[BOOK_COLUMNS].copy()

    # isbn lost its leading zeros in the source file
    df["isbn"] = df["isbn"].str.zfill(10)
    # isbn13 in the source has only 12 significant digits (check digit lost) -> rebuild it from isbn
    df["isbn13"] = df["isbn"].map(isbn10_to_isbn13, na_action="ignore")

    # "authors" lists every contributor of the edition (illustrator, translator, introduction …) without roles;
    # the main author is always first
    names = df.pop("authors").str.split(",").apply(lambda xs: [x.strip() for x in xs])
    df.insert(df.columns.get_loc("original_title") + 1, "author", names.str[0])
    df.insert(df.columns.get_loc("author") + 1, "contributors", names.str[1:].str.join(", ").replace("", None))

    df["original_title"] = df["original_title"].fillna(df["title"])
    # 76 titles are in Arabic / Japanese / Greek / Cyrillic script – the app is English, so it does not show them
    df.insert(df.columns.get_loc("title") + 1, "title_is_latin", df["title"].map(is_latin))
    df["original_publication_year"] = df["original_publication_year"].astype("Int64")
    df["language_code"] = df["language_code"].replace(dict.fromkeys(ENGLISH_CODES, "en"))
    # placeholder cover -> NaN, so a cover from Open Library can take its place
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


def add_covers(books: pd.DataFrame) -> pd.DataFrame:
    """Add cover_url: Open Library cover by ISBN, Goodreads image_url as fallback.

    Found cover ids are cached in data/cache/ol_covers.parquet (first run ~30 min, later only
    books still missing a cover are looked up). URLs by cover id have no rate limit, unlike URLs by ISBN.
    """
    cache_file = CACHE / "ol_covers.parquet"
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = (pd.read_parquet(cache_file) if cache_file.exists()
             else pd.DataFrame({"isbn": pd.Series(dtype="str"), "ol_cover_id": pd.Series(dtype="int64")}))

    todo = sorted(set(books["isbn"].dropna()) - set(cache["isbn"]))
    logger.info("cached covers: %d, ISBNs to look up: %d", len(cache), len(todo))
    n_batches = -(-len(todo) // COVER_BATCH)
    for i, start in enumerate(range(0, len(todo), COVER_BATCH), 1):
        batch = todo[start:start + COVER_BATCH]
        found = fetch_cover_ids(batch)
        if found:
            # save after every batch, so an interrupted run keeps what it already found
            new = pd.DataFrame({"isbn": list(found), "ol_cover_id": list(found.values())})
            cache = pd.concat([cache, new], ignore_index=True)
            cache.to_parquet(cache_file, index=False)
        logger.info("covers batch %d/%d: %d/%d found", i, n_batches, len(found), len(batch))
        time.sleep(REQUEST_DELAY)

    books = books.merge(cache, on="isbn", how="left")
    ol_url = books["ol_cover_id"].map(lambda x: COVER_URL.format(int(x)), na_action="ignore")
    books["cover_url"] = ol_url.fillna(books["image_url"])
    logger.info("covers: Open Library %d, Goodreads fallback %d, none %d",
                ol_url.notna().sum(), books["cover_url"].notna().sum() - ol_url.notna().sum(),
                books["cover_url"].isna().sum())
    return books.drop(columns=["ol_cover_id"])


def tag_genres(book_tags: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Votes per (goodreads_book_id, genre) from reader tags."""
    tag_to_genre = pd.DataFrame([(t, g) for g, ts in GENRES.items() for t in ts], columns=["tag_name", "genre"])
    df = book_tags[book_tags["count"] > 0].drop_duplicates().merge(tags, on="tag_id").merge(tag_to_genre, on="tag_name")
    return df.groupby(["goodreads_book_id", "genre"], as_index=False)["count"].sum()


def pick_genres(votes: pd.DataFrame) -> pd.Series:
    """Genres with at least GENRE_MIN_RELATIVE of the book's strongest genre votes, strongest first (max MAX_GENRES)."""
    votes = votes.assign(relative=votes["count"] / votes.groupby("goodreads_book_id")["count"].transform("max"))
    votes = votes[votes["relative"] >= GENRE_MIN_RELATIVE].sort_values(["goodreads_book_id", "count"],
                                                                       ascending=[True, False])
    return votes.groupby("goodreads_book_id")["genre"].apply(lambda g: list(g)[:MAX_GENRES])


def read_descriptions(zip_path: Path) -> dict[int, str]:
    """{goodreads_book_id: description} from the Goodreads XML files (plain-text extraction, no XML parser)."""
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in (n for n in zf.namelist() if n.endswith(".xml")):
            m = re.search(r"<book>.*?<description>(.*?)</description>", zf.read(name).decode("utf-8", "replace"), re.S)
            if m:
                desc = re.sub(r"<!\[CDATA\[|\]\]>", "", m.group(1))
                desc = re.sub(r"<br\s*/?>", "\n", html.unescape(desc))            # keep paragraphs
                desc = re.sub(r"<[^>]+>", "", desc)
                desc = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", desc)).strip()
                if desc:
                    out[int(Path(name).stem)] = desc
    return out


def add_genres_and_descriptions(books: pd.DataFrame) -> pd.DataFrame:
    """genres: list of 1–4 genres from reader tags; description: text from the Goodreads XML (NaN if missing)."""
    genres = pick_genres(tag_genres(pd.read_csv(RAW / "book_tags.csv"), pd.read_csv(RAW / "tags.csv")))
    books["genres"] = books["goodreads_book_id"].map(genres)
    books["description"] = books["goodreads_book_id"].map(read_descriptions(RAW / "books_xml.zip"))
    logger.info("genres: %d books, descriptions: %d books", books["genres"].notna().sum(), books["description"].notna().sum())
    return books


def detect_language(text: str) -> str:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0          # langdetect is random without a fixed seed
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


class Translator:
    """Translates text to English with the NLLB model; loaded only when something needs translating."""

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tokenizers = {}
        self.auto_tokenizer = AutoTokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)

    def __call__(self, text: str, src: str) -> str:
        if src not in self.tokenizers:
            self.tokenizers[src] = self.auto_tokenizer.from_pretrained(TRANSLATION_MODEL, src_lang=src)
        tok = self.tokenizers[src]
        paragraphs = []
        for paragraph in text.split("\n\n"):
            # sentence by sentence – the model handles up to ~512 tokens and translates short pieces better
            sentences = [s for s in re.split(r"(?<=[.!?…。])\s+", paragraph.strip()) if s]
            if not sentences:
                continue
            batch = tok(sentences, return_tensors="pt", padding=True, truncation=True, max_length=400)
            out = self.model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"), max_new_tokens=400)
            paragraphs.append(" ".join(tok.batch_decode(out, skip_special_tokens=True)))
        return "\n\n".join(paragraphs)


def add_english_descriptions(books: pd.DataFrame) -> pd.DataFrame:
    """description_en: the description in English – original when it is English, otherwise translated.

    Language and translation per book are cached in data/cache/translations.parquet, so the slow
    translation (~2 s per sentence on CPU) runs only once.
    """
    cache_file = CACHE / "translations.parquet"
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = (pd.read_parquet(cache_file) if cache_file.exists()
             else pd.DataFrame({"goodreads_book_id": pd.Series(dtype="int64"), "language": pd.Series(dtype="str"),
                                "description_en": pd.Series(dtype="str")}))
    todo = books[books["description"].notna() & ~books["goodreads_book_id"].isin(cache["goodreads_book_id"])]
    logger.info("cached descriptions: %d, to check: %d", len(cache), len(todo))

    rows, translator = [], None
    for i, (gid, text) in enumerate(zip(todo["goodreads_book_id"], todo["description"]), 1):
        lang = detect_language(text)
        english = text
        if lang != "en" and lang in NLLB_LANGS:
            translator = translator or Translator()
            english = translator(text, NLLB_LANGS[lang])
            logger.info("translated %s (%s)", gid, lang)
        rows.append({"goodreads_book_id": gid, "language": lang, "description_en": english})
        if lang != "en" or i == len(todo):             # save after every translation, so nothing is lost
            cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            cache.to_parquet(cache_file, index=False)
            rows = []

    books["description_en"] = books["goodreads_book_id"].map(cache.set_index("goodreads_book_id")["description_en"])
    logger.info("descriptions in English: %d (translated: %d)", books["description_en"].notna().sum(),
                (cache["language"].isin(NLLB_LANGS)).sum())
    return books


def fetch_author(name: str) -> dict:
    """Short Wikipedia summary of an author; empty fields when there is no page about a writer of that name."""
    row = {"author": name, "author_description": None, "author_bio": None, "author_url": None, "author_photo_url": None}
    url = WIKI_URL.format(requests.utils.quote(name.replace(" ", "_")))
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=WIKI_HEADERS, timeout=30)
        except requests.RequestException as e:
            logger.warning("author %s: %s", name, e)
            return row | {"checked": False}         # network error -> try again next run
        if resp.status_code != 429 and resp.status_code < 500:
            break
        time.sleep(float(resp.headers.get("retry-after", 2 ** attempt)))   # rate limited or server error -> wait
    if resp.ok:
        page = resp.json()
        text = f"{page.get('description', '')} {page.get('extract', '')[:300]}"
        if page.get("type") == "standard" and WRITER_WORDS.search(text):   # not a disambiguation page, is a writer
            row |= {"author_description": page.get("description"), "author_bio": page.get("extract"),
                    "author_url": page.get("content_urls", {}).get("desktop", {}).get("page"),
                    "author_photo_url": page.get("thumbnail", {}).get("source")}   # hotlinked from Wikimedia
    return row | {"checked": resp.ok or resp.status_code == 404}


def add_author_info(books: pd.DataFrame) -> pd.DataFrame:
    """Add author_description, author_bio, author_url and author_photo_url from Wikipedia.

    Every author is looked up only once – results (also "not found") are cached in data/cache/authors.parquet.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache_file = CACHE / "authors.parquet"
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = pd.read_parquet(cache_file) if cache_file.exists() else pd.DataFrame(columns=["author", "checked"])
    if "author_photo_url" not in cache.columns:        # cache from before photos were stored -> look up writers again
        cache["author_photo_url"] = None
        cache.loc[cache["author_bio"].notna(), "checked"] = False
    todo = sorted(set(books["author"]) - set(cache.loc[cache["checked"], "author"]))
    logger.info("cached authors: %d, authors to look up: %d", len(cache), len(todo))

    rows = []
    with ThreadPoolExecutor(WIKI_WORKERS) as pool:
        for i, row in enumerate(pool.map(fetch_author, todo), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(todo):         # save regularly, so an interrupted run keeps its results
                new = pd.DataFrame(rows)
                cache = pd.concat([cache[~cache["author"].isin(new["author"])], new], ignore_index=True)
                cache.to_parquet(cache_file, index=False)
                rows = []
                logger.info("authors: %d/%d looked up", i, len(todo))

    books = books.merge(cache.drop(columns="checked"), on="author", how="left")
    logger.info("author info from Wikipedia: %d of %d authors", cache["author_bio"].notna().sum(), books["author"].nunique())
    return books


def etl() -> None:
    books = clean_books(pd.read_csv(RAW / "books.csv"))
    interactions = clean_ratings(pd.read_csv(RAW / "ratings.csv"), books)
    validate(books, interactions)
    books = add_genres_and_descriptions(books)
    books = add_english_descriptions(books)
    books = add_author_info(books)
    books = add_covers(books)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    books.to_parquet(PROCESSED / "books.parquet", index=False)
    interactions.to_parquet(PROCESSED / "interactions.parquet", index=False)
    logger.info("saved %d books and %d interactions to %s", len(books), len(interactions), PROCESSED)


# ---------------------------------------------------------------- similar

def liked_matrix(interactions: pd.DataFrame, n_users: int, n_books: int) -> sp.csr_matrix:
    """User x book 0/1 matrix of liked books (0-based indexes = id - 1)."""
    liked = interactions[interactions["rating"] >= MIN_RATING]
    data = np.ones(len(liked), dtype=np.float32)
    return sp.csr_matrix((data, (liked["user_id"] - 1, liked["book_id"] - 1)), shape=(n_users, n_books))


def item_similarity(user_books: sp.csr_matrix, chunk: int = 1000) -> np.ndarray:
    """Book x book cosine similarity of shared readers: shared / sqrt(readers A * readers B).

    Dense 10k x 10k floats = 400 MB, built in chunks to save memory. Diagonal (book itself) = 0.
    """
    norms = np.sqrt(np.asarray(user_books.sum(axis=0)).ravel())
    cols = sp.csc_matrix(user_books.multiply(1 / np.maximum(norms, 1)).astype(np.float32))
    n_books = user_books.shape[1]
    sim = np.zeros((n_books, n_books), dtype=np.float32)
    for start in range(0, n_books, chunk):
        sim[:, start:start + chunk] = (cols.T @ cols[:, start:start + chunk]).toarray()
    np.fill_diagonal(sim, 0)
    return sim


def top_similar(sim: np.ndarray, k: int = TOP_SIMILAR, keep: np.ndarray | None = None) -> pd.DataFrame:
    """k most similar books for every book: book_id, rank, similar_book_id, score.

    keep (bool per book): only these books are listed and recommended. The similarity itself is
    computed from all books, so their readers still count.
    """
    keep = np.ones(sim.shape[0], dtype=bool) if keep is None else keep
    sim = np.where(keep[None, :], sim, -np.inf)[keep]          # rows = kept query books, other columns can't win
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    book_ids = np.flatnonzero(keep) + 1
    return pd.DataFrame({
        "book_id": np.repeat(book_ids, k).astype("int32"),
        "rank": np.tile(np.arange(1, k + 1), len(book_ids)).astype("int8"),
        "similar_book_id": (idx.ravel() + 1).astype("int32"),
        "score": np.take_along_axis(sim, idx, axis=1).ravel(),
    })


def similar() -> None:
    interactions = pd.read_parquet(PROCESSED / "interactions.parquet")
    books = pd.read_parquet(PROCESSED / "books.parquet").sort_values("book_id")
    n_users, n_books = int(interactions["user_id"].max()), int(interactions["book_id"].max())
    keep = books["title_is_latin"].to_numpy()
    df = top_similar(item_similarity(liked_matrix(interactions, n_users, n_books)), keep=keep)
    df.to_parquet(PROCESSED / "similar_books.parquet", index=False)
    logger.info("saved top %d similar books for %d books (%d with non-Latin titles left out)",
                TOP_SIMILAR, keep.sum(), (~keep).sum())


STEPS = {"ingest": ingest, "etl": etl, "similar": similar}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="*", default=list(STEPS), help=f"steps to run: {', '.join(STEPS)}")
    parser.add_argument("--force", action="store_true", help="ingest: download again")
    args = parser.parse_args()
    if unknown := set(args.steps) - set(STEPS):
        parser.error(f"unknown steps: {', '.join(sorted(unknown))}")
    for step in args.steps:
        logger.info("=== %s", step)
        ingest(args.force) if step == "ingest" else STEPS[step]()
