"""Book recommender app: pick a book or an author, get similar books (item-item, see pipeline.py).

- Book   -> up to 50 most similar books, paged.
- Author -> the author's books + up to 50 books by other authors that the author's readers liked, paged
            (similarities to all of the author's books summed up).

Usage:
    streamlit run app/app.py        (from the repo root, so .streamlit/config.toml is used)

Reads books.parquet and similar_books.parquet from DATA_PATH (default: <repo>/data/processed).
After IDLE_SECONDS without any activity the page moves to static/expired.html, which closes the
connection to the server – Cloud Run bills Streamlit for as long as the connection is open.
"""

import os
import re
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]   # works no matter where the app is started from
DATA_PATH = Path(os.environ.get("DATA_PATH", REPO_ROOT / "data" / "processed"))
HEADER_PHOTO = Path(__file__).parent / "static" / "header.jpg"   # st.image, works without static serving
N_RECOMMENDATIONS = 10            # per page
N_LIKED = 50                      # recommendations to page through (pipeline keeps TOP_SIMILAR = 50 per book)
N_AUTHOR_BOOKS = 5
PAGER_NUMBERS = 5                 # page numbers shown in the author pager, the rest as dots
N_COLUMNS = 5
IDLE_SECONDS = 120

# --- about the app: update APP_VERSION and APP_UPDATED with every release
APP_VERSION = "1.0.0"
APP_UPDATED = "2026-10-24"
APP_AUTHOR = "Martin Bílek"
REPO_URL = "https://github.com/martincho48/book_recommender"

Choice = tuple[str, int | str]   # ("book", book_id) or ("author", name)

# --- look: all colors and fonts in one place, turned into CSS by apply_styles()
COLORS = {
    "page": "#eef0f3",          # light gray page
    "card": "#dfe4ea",          # sections: a darker gray-blue
    "card_shadow": "rgba(30, 41, 59, 0.18)",
    "book_card": "#f6f7f9",     # one book inside a section: almost white
    "book_shadow": "rgba(30, 41, 59, 0.12)",
    "text": "#1f2937",
    "muted": "#5b6474",
    "accent": "#b4532a",        # warm book-leather brown for headings and links
    "author": "#374151",        # author name: darker than the muted text
    "genre": "#3f7d78",         # genres: muted teal, goes with the brown accent
    "no_cover": "#c7ced8",
}
# logo: white open book on a rounded square in the accent color
LOGO_SVG = f"""<svg class="logo" viewBox="0 0 48 48" aria-hidden="true">
  <rect width="48" height="48" rx="12" fill="{COLORS["accent"]}"/>
  <path d="M9 15c5-2.2 10-2.2 14 1v19c-4-2.8-9-2.8-14-1z M39 15c-5-2.2-10-2.2-14 1v19c4-2.8 9-2.8 14-1z" fill="#fff"/>
</svg>"""
FONTS = {
    "import": "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Merriweather:wght@700&display=swap",
    "body": "'Inter', sans-serif",
    "heading": "'Merriweather', serif",
    "title_size": "2.4rem",
    "subheader_size": "1.5rem",
    "section_size": "1.15rem",
    "body_size": "0.95rem",
    "caption_size": "0.82rem",
    "card_title_size": "0.88rem",   # book title in a book card
    "card_meta_size": "0.76rem",    # author and genres in a book card
}
COVER_HEIGHT = 220   # px, all covers in book cards have this height
CARDS = ["search", "selected", "author_books", "results", "about"]    # keys of the st.container sections
SEARCH_KEY = "query"                          # session state key of the search box (Reset clears it)
# light from the top left -> shadow only on the right and bottom edges (x, y, blur, spread)
SHADOW = "6px 8px 14px -4px"
BOOK_SHADOW = "3px 4px 8px -2px"


def data_version() -> tuple[float, float]:
    """Modification times of the data files – part of the cache key, so new pipeline output is picked up."""
    return tuple((DATA_PATH / name).stat().st_mtime for name in ("books.parquet", "similar_books.parquet"))


@st.cache_data
def load_data(version: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Books (sorted by popularity: book_id 1 = most rated) and the precomputed similar books.
    Cached in memory; `version` changes when the pipeline rewrites the files, which reloads them."""
    books = pd.read_parquet(DATA_PATH / "books.parquet").sort_values("book_id").set_index("book_id")
    books = books[books["title_is_latin"]]          # the app is English – no Arabic / Japanese / … titles
    similar = pd.read_parquet(DATA_PATH / "similar_books.parquet")
    return books, similar


@st.cache_data
def search_options(_books: pd.DataFrame) -> tuple[list[Choice], dict[str, int]]:
    """All books and authors, by popularity; an author comes right before their most popular book.
    (The leading underscore tells Streamlit not to hash the books table for the cache.)"""
    authors = _books.reset_index().groupby("author")["book_id"].agg(["min", "size"])
    ranked = [(book_id, 1, ("book", book_id)) for book_id in _books.index]
    ranked += [(row["min"], 0, ("author", name)) for name, row in authors.iterrows()]
    return [choice for *_, choice in sorted(ranked)], authors["size"].to_dict()


def choice_label(books: pd.DataFrame, n_books: dict[str, int], choice: Choice) -> str:
    """Text shown in the search box."""
    kind, value = choice
    if kind == "author":
        return f"✍️ {value} ({n_books[value]} {'book' if n_books[value] == 1 else 'books'})"
    book = books.loc[value]
    return f"📖 {book.title} – {book.author}"


def reset_search() -> None:
    st.session_state[SEARCH_KEY] = None


def pick(books: pd.DataFrame) -> Choice | None:
    """Search box (as wide as the column it is in) that suggests only books and authors in the catalogue, with a Reset button."""
    options, n_books = search_options(books)
    choice = st.selectbox(
        "Book or author",
        options=options,
        format_func=lambda choice: choice_label(books, n_books, choice),
        index=None,
        placeholder="Start typing a title or an author…",
        filter_mode="contains",
        label_visibility="collapsed",
        key=SEARCH_KEY,
    )
    st.button("Reset", icon=":material/restart_alt:", on_click=reset_search, disabled=choice is None)
    return choice


def similar_to_book(similar: pd.DataFrame, books: pd.DataFrame, book_id: int, n: int = N_LIKED) -> pd.DataFrame:
    """The n most similar books, best first."""
    ids = similar.loc[similar.book_id == book_id].nsmallest(n, "rank").similar_book_id
    return books.loc[ids]


def author_books(books: pd.DataFrame, author: str) -> pd.DataFrame:
    """All books of the author, most popular first."""
    return books[books.author == author]


def similar_to_author(similar: pd.DataFrame, books: pd.DataFrame, author: str, n: int = N_LIKED) -> pd.DataFrame:
    """Books by other authors, ranked by summed similarity to all of the author's books."""
    own = books.index[books.author == author]
    scores = similar.loc[similar.book_id.isin(own)].groupby("similar_book_id")["score"].sum()
    scores = scores[books.loc[scores.index, "author"] != author]
    return books.loc[scores.nlargest(n).index]


def show_cover(url: str | None) -> None:
    """Cover with the same height everywhere (COVER_HEIGHT), styled by the .cover CSS class."""
    if pd.notna(url):
        st.markdown(f"<img class='cover' src='{escape(url, quote=True)}' alt=''>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='cover no-cover'>No cover</div>", unsafe_allow_html=True)


def genre_text(genres) -> str:
    return " · ".join(g.replace("-", " ") for g in genres) if genres is not None else ""


def show_description(book: pd.Series) -> None:
    """Small link-like button; the English description (translated when needed) opens in a floating panel."""
    if pd.notna(book.description_en):
        with st.popover("Description", type="tertiary", icon=":material/menu_book:"):
            st.markdown(f"**{book.title}**")
            st.write(book.description_en)


def author_photo_html(book: pd.Series, height: int) -> str:
    """Author portrait (3:4, like the book covers) hotlinked from Wikimedia, or a person icon when there is none."""
    width = height * 3 // 4
    url = book.get("author_photo_url")
    if pd.notna(url):
        return f"<img class='author-photo' src='{escape(url, quote=True)}' alt='' style='width:{width}px;height:{height}px'>"
    return f"<div class='author-photo no-photo' style='width:{width}px;height:{height}px;font-size:{height // 3}px'>👤</div>"


def author_html(book: pd.Series) -> str:
    """Author name in a book card; on mouse over a bubble with photo, description and the start of the bio."""
    name = f"<div class='book-author book-line'>{escape(book.author)}</div>"
    if pd.isna(book.author_bio) and pd.isna(book.author_description):
        return name
    about = f"<i>{escape(book.author_description)}</i>" if pd.notna(book.author_description) else ""
    bio = f"<div class='tip-bio'>{escape(book.author_bio)}</div>" if pd.notna(book.author_bio) else ""
    return (f"<div class='author-hover'>{name}<div class='author-tip'>"
            f"<div class='author-head'>{author_photo_html(book, 64)}<div><b>{escape(book.author)}</b><br>{about}</div></div>"
            f"{bio}<div class='tip-source'>Source: Wikipedia</div></div></div>")


def show_author_bio(book: pd.Series) -> None:
    """Author bio from Wikipedia (text and photo under CC BY-SA, so the source is linked)."""
    st.write(book.author_bio)
    st.caption(f"Source: [Wikipedia]({book.author_url})")


def show_author_info(book: pd.Series) -> None:
    """Small link-like button; the author photo and bio open in a floating panel."""
    if pd.notna(book.author_bio):
        with st.popover("About the author", type="tertiary", icon=":material/person:"):
            st.markdown(f"<div class='author-head'>{author_photo_html(book, 96)}<b>{escape(book.author)}</b></div>",
                        unsafe_allow_html=True)
            show_author_bio(book)


def show_paged_books(heading: str, books: pd.DataFrame, key: str, per_page: int) -> None:
    """Books per_page at a time; heading with [1–5 / 80] and a pager ‹ 1 2 3 4 5 • • • › to page through the rest."""
    key = re.sub(r"[^A-Za-z0-9]+", "_", key)         # ends up in CSS classes
    page_key = f"page_{key}"                          # remembered per section and author while the session lasts
    n_pages = max(-(-len(books) // per_page), 1)
    page = min(st.session_state.get(page_key, 0), n_pages - 1)
    start, end = page * per_page, min((page + 1) * per_page, len(books))

    def go(to: int) -> None:
        st.session_state[page_key] = to

    title, pager = st.columns([3, 2], vertical_alignment="center")
    with title:
        shown = f" [{start + 1}–{end} / {len(books)}]" if n_pages > 1 else ""
        st.markdown(f"#### {escape(heading)}{shown}")
    if n_pages > 1:
        first = min(max(page - PAGER_NUMBERS // 2, 0), max(n_pages - PAGER_NUMBERS, 0))   # numbers shown around page
        last = min(first + PAGER_NUMBERS, n_pages)

        def dots(hidden: int) -> None:                 # pages without a number, max 4 dots
            if hidden:
                st.markdown(f"<span class='pager-dots'>{' '.join('•' * min(hidden, 4))}</span>", unsafe_allow_html=True)

        with pager, st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center",
                                 gap="small", key=f"paging_{key}"):
            st.button("", icon=":material/chevron_left:", help="Previous books", on_click=go, args=(page - 1,),
                      disabled=page == 0, key=f"prev_{key}")
            dots(first)
            for n in range(first, last):
                st.button(str(n + 1), type="tertiary", on_click=go, args=(n,),
                          key=f"{'pagecur' if n == page else 'pagenum'}_{key}_{n}")
            dots(n_pages - last)
            st.button("", icon=":material/chevron_right:", help="More books", on_click=go, args=(page + 1,),
                      disabled=page == n_pages - 1, key=f"next_{key}")
    show_books(None, books.iloc[start:end], key=f"{key}_{page}")


def show_selected_book(book: pd.Series) -> None:
    left, right = st.columns([1, 4])
    with left:
        show_cover(book.cover_url)
    with right:
        st.subheader(book.title)
        about = f" – <i>{escape(book.author_description)}</i>" if pd.notna(book.author_description) else ""
        year = f" | first published {book.original_publication_year}" if pd.notna(book.original_publication_year) else ""
        st.markdown(f"<div class='selected-author'><span class='book-author'>{escape(book.author)}</span>{about}</div>"
                    f"<div class='selected-meta'><span class='book-genres'>{escape(genre_text(book.genres))}</span>{year}</div>",
                    unsafe_allow_html=True)
        with st.container(horizontal=True, gap="medium"):
            show_description(book)
            show_author_info(book)


def show_selected_author(author_book: pd.Series) -> None:
    """Same layout as a selected book: photo where the cover is, name, description and bio on the right.
    author_book is any book row of the author (author info is the same in all of them)."""
    left, right = st.columns([1, 4])
    with left:
        st.markdown(author_photo_html(author_book, COVER_HEIGHT), unsafe_allow_html=True)
    with right:
        st.subheader(author_book.author)
        if pd.notna(author_book.author_description):
            st.markdown(f"<div class='selected-meta'>{escape(author_book.author_description)}</div>",
                        unsafe_allow_html=True)
        if pd.notna(author_book.author_bio):
            show_author_bio(author_book)


def book_card(book: pd.Series, key: str) -> None:
    """One book in its own small card: fixed-height cover, title (max 2 lines), author (bubble with info on mouse over),
    genres, description on click."""
    with st.container(key=f"book_{key}"):
        show_cover(book.cover_url)
        st.markdown(f"<div class='book-title' title='{escape(book.title, quote=True)}'>{escape(book.title)}</div>"
                    f"{author_html(book)}"
                    f"<div class='book-genres book-line'>{escape(genre_text(book.genres))}</div>",
                    unsafe_allow_html=True)
        show_description(book)


def show_books(heading: str | None, books: pd.DataFrame, key: str | None = None) -> None:
    """Book cards in rows of N_COLUMNS – every row starts aligned. heading=None: the caller draws its own."""
    if heading:
        st.markdown(f"#### {heading}")
    key = key or re.sub(r"[^A-Za-z0-9]+", "_", heading)
    for start in range(0, len(books), N_COLUMNS):
        row = books.iloc[start:start + N_COLUMNS]
        for column, (book_id, book) in zip(st.columns(N_COLUMNS), row.iterrows()):
            with column:
                book_card(book, key=f"{key}_{book_id}")   # key ends up as a CSS class


def fonts_css() -> str:
    """All fonts of the page – families and sizes come from FONTS."""
    return f"""
        @import url('{FONTS["import"]}');
        .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp [data-baseweb="select"] div {{
            font-family: {FONTS["body"]}; font-size: {FONTS["body_size"]}; color: {COLORS["text"]};
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: {FONTS["heading"]}; color: {COLORS["text"]}; }}
        .stApp h1 {{ font-size: {FONTS["title_size"]}; }}
        .stApp h3 {{ font-size: {FONTS["subheader_size"]}; }}
        .stApp h4 {{ font-size: {FONTS["section_size"]}; color: {COLORS["accent"]}; }}
        .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {{
            font-size: {FONTS["caption_size"]}; color: {COLORS["muted"]};
        }}
        .stApp a {{ color: {COLORS["accent"]}; }}
        .book-title {{ font-family: {FONTS["body"]}; font-weight: 700; font-size: {FONTS["card_title_size"]}; color: {COLORS["text"]}; }}
        .book-line {{ font-family: {FONTS["body"]}; font-size: {FONTS["card_meta_size"]}; }}
        .book-author {{ color: {COLORS["author"]}; font-weight: 600; }}
        .book-genres {{ color: {COLORS["genre"]}; }}
        .selected-author, .selected-meta {{ font-family: {FONTS["body"]}; }}
        .author-head {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 0.6rem; }}
        .author-head h3 {{ margin: 0 0 0.2rem; padding: 0; }}
        .author-photo {{   /* !important: wins over the general .stApp img rule, also in popovers outside .stApp */
            border-radius: 6px !important; object-fit: cover; object-position: top; flex-shrink: 0;
            margin-bottom: 0.9rem;   /* same gap below as under the book covers */
            box-shadow: {BOOK_SHADOW} {COLORS["book_shadow"]};
        }}
        .no-photo {{ display: flex; align-items: center; justify-content: center; background: {COLORS["no_cover"]}; }}
        .selected-author {{ font-size: {FONTS["body_size"]}; color: {COLORS["muted"]}; margin-bottom: 0.3rem; }}
        .selected-meta {{ font-size: {FONTS["caption_size"]}; color: {COLORS["muted"]}; margin-bottom: 0.6rem; }}
        /* Description / About the author: small brown links with an icon, everywhere */
        /* pager ‹ 1 2 3 4 5 • • • ›: small square arrows, muted numbers, current page in accent */
        [class*="st-key-paging"] button {{ min-height: 0; padding: 0.15rem 0.35rem; color: {COLORS["accent"]}; }}
        [class*="st-key-paging"] [class*="st-key-pagenum"] button p {{ color: {COLORS["muted"]}; }}
        [class*="st-key-paging"] [class*="st-key-pagecur"] button p {{ color: {COLORS["accent"]}; font-weight: 700; }}
        .pager-dots {{ color: {COLORS["muted"]}; letter-spacing: 0.1rem; }}
        [data-testid="stPopover"] button, [data-testid="stPopover"] button p {{
            font-size: {FONTS["caption_size"]}; color: {COLORS["accent"]};
        }}
        [class*="st-key-book_"] [data-testid="stPopover"] button, [class*="st-key-book_"] [data-testid="stPopover"] button p {{
            font-size: {FONTS["card_meta_size"]};
        }}
    """


def cards_css() -> str:
    """Light gray page; the sections (st.container with a key from CARDS) are darker rounded cards with a shadow."""
    cards = ", ".join(f".st-key-{key}" for key in CARDS)
    return f"""
        .stApp {{ background: {COLORS["page"]}; }}
        {cards} {{
            background: {COLORS["card"]};
            border-radius: 16px;
            box-shadow: {SHADOW} {COLORS["card_shadow"]};
            padding: 1.4rem 1.6rem;
            margin-bottom: 0.6rem;
        }}
        .stApp [data-testid="stExpander"] details {{
            background: rgba(255, 255, 255, 0.55); border: none; border-radius: 10px;
        }}
        .stApp img {{ border-radius: 6px; }}

        /* one book = small card with fixed-size parts, so all cards in a row have the same height */
        [class*="st-key-book_"] {{
            background: {COLORS["book_card"]};
            border-radius: 12px;
            box-shadow: {BOOK_SHADOW} {COLORS["book_shadow"]};
            padding: 0.9rem 0.9rem 0.4rem;
            min-height: {COVER_HEIGHT + 120}px;
            gap: 0.3rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        [class*="st-key-book_"]:hover {{                   /* the card under the mouse comes out a little */
            transform: translateY(-4px) scale(1.03); position: relative; z-index: 5;   /* above its neighbours */
            box-shadow: 6px 10px 20px -4px {COLORS["card_shadow"]};
        }}
        /* author bubble in a book card, shown on mouse over the name */
        .author-hover {{ position: relative; }}
        .author-tip {{
            display: none; position: absolute; top: 100%; left: 0; z-index: 10; width: 320px; margin-top: 0.3rem;
            padding: 0.8rem 0.9rem; background: {COLORS["book_card"]}; border-radius: 10px;
            box-shadow: 6px 10px 24px -4px {COLORS["card_shadow"]};
            font-family: {FONTS["body"]}; font-size: {FONTS["card_meta_size"]}; color: {COLORS["text"]}; line-height: 1.45;
        }}
        .author-hover:hover .author-tip {{ display: block; }}
        [data-testid="stColumn"]:last-child .author-tip {{ left: auto; right: 0; }}   /* last column: open to the left */
        .author-hover .book-author {{ cursor: help; }}
        .author-tip .author-head {{ gap: 0.7rem; margin-bottom: 0.5rem; color: {COLORS["muted"]}; }}
        .author-tip .author-head b {{ color: {COLORS["text"]}; }}
        .author-tip .author-photo {{ margin-bottom: 0; }}
        .tip-bio {{ overflow: hidden; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 6; }}
        .tip-source {{ margin-top: 0.4rem; color: {COLORS["muted"]}; font-size: {FONTS["caption_size"]}; }}
        .cover {{
            display: block; height: {COVER_HEIGHT}px; width: auto; max-width: 100%; margin: 0 0 0.9rem;   /* left = aligned with the text */
            object-fit: contain; border-radius: 6px; box-shadow: {BOOK_SHADOW} {COLORS["book_shadow"]};
        }}
        .no-cover {{
            width: {COVER_HEIGHT * 2 // 3}px; display: flex; align-items: center; justify-content: center;
            background: {COLORS["no_cover"]}; color: {COLORS["muted"]};
        }}
        .book-title {{                                   /* max 2 lines, longer titles end with … */
            overflow: hidden; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2;
            height: 2.6em; line-height: 1.3;
        }}
        .stApp .app-title {{ padding-top: 0; }}
        .st-key-brand {{ width: fit-content !important; max-width: 100%; }}
        /* the suggestions may be wider than the search box, so long titles fit */
        [data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {{
            min-width: min(780px, 90vw) !important;
        }}
        .logo {{ width: 2.6rem; height: 2.6rem; margin-right: 0.8rem; vertical-align: -0.35rem; filter: drop-shadow({BOOK_SHADOW} {COLORS["book_shadow"]}); }}
        .app-title [data-testid="stHeaderActionElements"] {{ display: none; }}   /* no link icon next to the title */
        .st-key-header_photo img {{                       /* !important: wins over the general .stApp img rule */
            display: block; width: 100%; height: 230px; object-fit: cover !important; object-position: center 60%;
            border-radius: 12px !important; box-shadow: {BOOK_SHADOW} {COLORS["book_shadow"]};
        }}
        .book-line {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.5; }}
        .book-title + .book-line {{ margin-top: 0.15rem; }}
    """


def apply_styles() -> None:
    st.markdown(f"<style>{fonts_css()}{cards_css()}</style>", unsafe_allow_html=True)


def idle_guard(seconds: int = IDLE_SECONDS) -> None:
    """After `seconds` without clicks, typing, mouse moves or scrolling, leave the page to close the connection."""
    st.iframe(f"""
        <script>
          const page = window.parent;
          let timer;
          const reset = () => {{
            clearTimeout(timer);
            timer = setTimeout(() => page.location.replace("/app/static/expired.html"), {seconds} * 1000);
          }};
          ["click", "keydown", "mousemove", "scroll", "touchstart"].forEach(e => page.document.addEventListener(e, reset));
          reset();
        </script>""", height=1)


def show_about() -> None:
    """Card at the bottom: how the app works, data sources, author, version and date."""
    with st.container(key="about"):
        st.markdown("#### About the app")
        st.markdown(
            "Recommendations are based on **collaborative filtering**. Data based on 6M ratings from "
            "[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) (CC BY-SA 4.0), covers from "
            "[Open Library](https://openlibrary.org), author info from [Wikipedia](https://en.wikipedia.org), "
            "header photo by [Eldem Ozdemir](https://unsplash.com/@eldemozd) on [Unsplash](https://unsplash.com).")
        st.caption(f"© {date.today().year} {APP_AUTHOR} · version {APP_VERSION} · updated {APP_UPDATED} · "
                   f"[source code on GitHub]({REPO_URL})")


def main() -> None:
    st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
    apply_styles()
    idle_guard()
    books, similar = load_data(data_version())

    with st.container(key="search"):
        left, right = st.columns([2, 3], gap="large")
        with left, st.container(key="brand"):          # as wide as the title, so the search box ends with it
            st.markdown(f"<h1 class='app-title'>{LOGO_SVG}Book Recommender</h1>", unsafe_allow_html=True)
            st.caption("Pick a book or an author and see what their readers also liked.")
            choice = pick(books)
        if HEADER_PHOTO.exists():                     # optional – the app works without it
            with right, st.container(key="header_photo"):
                st.image(str(HEADER_PHOTO), width="stretch")
    if choice is not None:
        show_results(choice, books, similar)
    show_about()


def show_results(choice: Choice, books: pd.DataFrame, similar: pd.DataFrame) -> None:
    kind, value = choice
    if kind == "book":
        with st.container(key="selected"):
            show_selected_book(books.loc[value])
        with st.container(key="results"):
            show_paged_books("Readers who liked it also liked", similar_to_book(similar, books, value),
                             key=f"liked_book_{value}", per_page=N_RECOMMENDATIONS)
    else:
        own = author_books(books, value)
        with st.container(key="selected"):
            show_selected_author(own.iloc[0])
        with st.container(key="author_books"):
            show_paged_books(f"{'Book' if len(own) == 1 else 'Books'} by {value}", own,
                             key=f"author_{value}", per_page=N_AUTHOR_BOOKS)
        with st.container(key="results"):
            show_paged_books(f"Readers of {value} also liked", similar_to_author(similar, books, value),
                             key=f"liked_{value}", per_page=N_RECOMMENDATIONS)


main()
