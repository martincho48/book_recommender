import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pipeline import (clean_books, clean_ratings, detect_language, is_latin, isbn10_to_isbn13, item_similarity, pick_genres, tag_genres,
                      top_similar, validate)


def make_books() -> pd.DataFrame:
    return pd.DataFrame({
        "book_id": [1, 2], "goodreads_book_id": [10, 20], "title": ["A", "B"], "original_title": [None, "B"],
        "authors": ["Vladimir Nabokov, Craig Raine", "y"], "original_publication_year": [2008.0, None], "language_code": ["en-US", "fre"],
        "isbn": ["439023483", None], "isbn13": [9.78043902348e12, None],
        "image_url": ["https://img/1.jpg", "https://s.gr-assets.com/assets/nophoto/book/111x148.png"],
    })


def test_isbn13_check_digit():
    assert isbn10_to_isbn13("0439023483") == "9780439023481"   # The Hunger Games
    assert isbn10_to_isbn13("014028009X") == "9780140280098"   # ISBN-10 ending with X


def test_clean_books_fixes_formats():
    books = clean_books(make_books())
    assert books.loc[0, "isbn"] == "0439023483"                # leading zero restored
    assert books.loc[0, "isbn13"] == "9780439023481"           # rebuilt, not the rounded float
    assert books.loc[0, "original_title"] == "A"               # filled from title
    assert books.loc[0, "language_code"] == "en"
    assert pd.isna(books.loc[1, "image_url"])                  # placeholder cover removed
    assert books.loc[0, "author"] == "Vladimir Nabokov"        # main author split from other contributors
    assert books.loc[0, "contributors"] == "Craig Raine"
    assert pd.isna(books.loc[1, "contributors"])
    assert "authors" not in books.columns
    assert len(books) == 2                                     # books with NaN are kept


def test_clean_ratings_drops_invalid_rows():
    ratings = pd.DataFrame({"user_id": [1, 1, 2, 3, 4], "book_id": [1, 1, 2, 99, 1], "rating": [3, 5, 6, 4, None]})
    clean = clean_ratings(ratings, clean_books(make_books()))
    assert clean.values.tolist() == [[1, 1, 5]]                # duplicate keeps last, rating 6, unknown book, NaN gone


def test_validate_catches_book_without_ratings():
    books = clean_books(make_books())
    with pytest.raises(AssertionError, match="book without any interaction"):
        validate(books, pd.DataFrame({"user_id": [1], "book_id": [1], "rating": [5]}))


def test_tag_genres_merge_variants_and_ignore_shelves():
    tags = pd.DataFrame({"tag_id": [1, 2, 3], "tag_name": ["sci-fi", "scifi", "to-read"]})
    book_tags = pd.DataFrame({"goodreads_book_id": [7, 7, 7], "tag_id": [1, 2, 3], "count": [30, 20, 999]})
    assert tag_genres(book_tags, tags).set_index("genre")["count"].to_dict() == {"science-fiction": 50}


def test_pick_genres_keeps_strong_genres_only():
    votes = pd.DataFrame({"goodreads_book_id": [7] * 3, "genre": ["fantasy", "romance", "horror"], "count": [100, 40, 10]})
    assert pick_genres(votes)[7] == ["fantasy", "romance"]     # horror has only 10 % of the top genre


def test_item_similarity_is_cosine_of_shared_readers():
    # readers: book 0 = {0, 1, 2}, book 1 = {0, 1}, book 2 = {3}
    users = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32))
    sim = item_similarity(users)
    assert sim[0, 1] == pytest.approx(2 / np.sqrt(3 * 2))      # 0.82
    assert sim[0, 2] == 0
    assert sim[0, 0] == 0                                      # a book is not similar to itself


def test_top_similar_is_sorted_and_1_based():
    sim = np.array([[0, 0.9, 0.1], [0.9, 0, 0.5], [0.1, 0.5, 0]], dtype=np.float32)
    top = top_similar(sim, k=2)
    first = top[top.book_id == 1]
    assert first.similar_book_id.tolist() == [2, 3]
    assert first["rank"].tolist() == [1, 2]


def test_detect_language():
    assert detect_language("Guy Montag is a fireman and his job is to burn books.") == "en"
    assert detect_language("Guy Montag es un bombero y el trabajo de un bombero es quemar libros.") == "es"


def test_is_latin_checks_the_script():
    assert is_latin("Les Misérables (Vol. 1)")                 # accents, digits and punctuation are fine
    assert not is_latin("ساق البامبو")                          # Arabic
    assert not is_latin("NARUTO -ナルト- 巻ノ四十三")             # mixed Latin + Japanese


def test_top_similar_leaves_out_books_not_kept():
    sim = np.array([[0, 0.9, 0.1], [0.9, 0, 0.5], [0.1, 0.5, 0]], dtype=np.float32)
    top = top_similar(sim, k=1, keep=np.array([True, False, True]))
    assert top.book_id.tolist() == [1, 3]                      # book 2 is not listed …
    assert 2 not in top.similar_book_id.tolist()               # … and never recommended
