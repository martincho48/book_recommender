"""Smoke test of the app with the real data (data/processed): it starts and shows a book and an author without errors.
Runs locally with `pytest` and in GitHub Actions before every deployment."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).with_name("app.py")
DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
pytestmark = pytest.mark.skipif(not (DATA / "books.parquet").exists(), reason="no data – run pipeline/pipeline.py first")


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    return at


@pytest.mark.parametrize("kind", ["book", "author"])
def test_pick(app: AppTest, kind: str) -> None:
    box = app.selectbox[0]
    index = next(i for i, label in enumerate(box.options) if label.startswith("📖" if kind == "book" else "✍"))
    box.select_index(index).run()
    assert not app.exception
    assert app.selectbox[0].value[0] == kind
