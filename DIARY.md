# Deník projektu – Book Recommender

## Repo a venv
- Repo: `D:\03_Coding\project_repo\book_recommender` → GitHub `martincho48/book_recommender`
- Venv: `C:\Users\Martin\uv_venv\uv_book_recommender` (uv, Python 3.10) – mimo repo; jako Jupyter kernel „Python (book_recommender)“

## Jak spustit (Git Bash, z kořene repa)
```bash
source /c/Users/Martin/uv_venv/uv_book_recommender/Scripts/activate
python prod/pipeline.py                                        # data (vše z cache ~1 min)
streamlit run prod/app.py --server.enableStaticServing true    # app → http://localhost:8501
pytest                                                         # testy pipeline
```

## Otevřené
- Neověřeno: hlídání nečinnosti (JS v `idle_guard`) – zkusit v prohlížeči 2 min nic nedělat.
- Repo zatím nepushnuté na GitHub (odkaz v About ještě nefunguje); commity dělá uživatel sám.
- Dockerfile, nasazení na Cloud Run, pojistky (max-instances, budget, kill switch), `browser.gatherUsageStats=false`, GitHub Actions.

## Záznamy

### 2026-10-24
- **Data:** goodbooks-10k (Goodreads API zavřené, Open Library nemá ID uživatelů); Open Library jen na obálky. Starý projekt na Book-Crossing (`d:\03_Coding\Portfolio\...\Book Recommendation System`) – horší data, převzít z něj lze Flask app + Dockerfile.
- **Plán:** 1) lokálně: pipeline → modely → app → Docker; 2) GCP: Cloud Storage + Cloud Run, GitHub Actions; 3) volitelně měsíční aktualizace obálek.
- **Pipeline** (`prod/pipeline.py`): ingest → etl (čištění, opravené ISBN, `authors` → `author` + `contributors`, validace, obálky z Open Library s cache – první běh ~28 min) → podobné knihy (item-item, top 20) → `data/processed/` = jen 3 soubory.
- **Research** (`research/experiments.ipynb`): split 80/10/10 na uživatele z hodnocení ≥ 4, hodnocení 1–3 se nikdy nedoporučují; modely popularita, item-item, ALS, two-tower v1 (ID) a v2 (+ žánry z tagů, autor, rok, jazyk, vektor popisu z lokálního HF modelu all-MiniLM-L6-v2); výstupy v `research/output/`: `results.xlsx`, `data_*.parquet`, `model_*.pt`.
- Hlavní metrika = book_hit_rate@10 (scénář aplikace). Test: **item-item 0.344**, two-tower 0.294, v2 0.292, popularita 0.289, ALS 0.285. Two-tower vede jen v doporučení uživateli (Recall@10 0.212 vs. 0.175) → **aplikace použije item-item**.
- Úklid struktury: z `src/` (13 skriptů), `docs/`, `artifacts/` → `prod/` (produkce), `research/` (2 notebooky), `prod/test_pipeline.py`. Výsledky ověřeny – stejná čísla.
- **Aplikace:** rozhodnuto pro **Streamlit** na Cloud Run (platí se za dobu otevřeného spojení, ne za dotaz). Pojistky při nasazení: `--max-instances=1`, `--timeout=300`, budget 5 USD, kill switch (budget → Pub/Sub → funkce odebere veřejný přístup `allUsers`). Nejhorší měsíc ~5–7 USD.
- **Streamlit app** (`prod/app.py`; funkce `load_data`, `search_options`, `pick`, `similar_to_book`, `similar_to_author`, `book_card`, `show_*`, `apply_styles`, `idle_guard`, `show_about`): našeptávač jen knih/autorů z katalogu, 10 podobných knih s obálkami, po 2 min nečinnosti přesměrování na `prod/static/expired.html` (zavře spojení).
- App: jedno vyhledávací pole pro knihy i autory (13 888 položek). Autor → top 5 jeho knih + 10 knih jiných autorů (součet item-item podobností ke všem jeho knihám). Algoritmus stejný, pipeline beze změny.
- Žánry (z tagů) a popisy (Goodreads XML) přesunuty do pipeline → `books.parquet` sloupce `genres`, `description`; notebook je bere odtud. App: žánry u knih, popis v rozklikávacím poli.
- Info o autorech z Wikipedie (REST API, bez LLM): pipeline `add_author_info`, cache `data/cache/authors.parquet` (stahuje se jednou, ~12 min; 2 paralelní dotazy + opakování při 429). Nalezeno 2 876 z 3 888 autorů. App: popis u autora + rozklikávací „About the author“ s odkazem na zdroj.
- App layout: světle šedá stránka, 3 karty (`st.container(key=...)`: search, selected, results) se zaoblenými rohy, stínem a tmavší výplní; barvy a fonty v `COLORS` / `FONTS`, CSS z `fonts_css()` + `cards_css()` → `apply_styles()`. Fonty Inter (text) + Merriweather (nadpisy). Každá kniha ve vlastní kartě (`book_card`), řádky po 5, obálky stejně vysoké, název max. 2 řádky, popis v plovoucím okně (`st.popover`).
- App: stíny jen vpravo/dole (světlo zleva), vyhledávání na polovinu šířky, Reset pod ním; karta „About the app“ (zdroje dat, © autor, verze `APP_VERSION`, `APP_UPDATED` – při vydání ručně přepsat, bez data dat). Playwright (dev) na kontrolu vzhledu v Chrome.
- Popisy: jen ~200 neanglických (2 %) → překlad do EN lokálním modelem NLLB (`facebook/nllb-200-distilled-600M`), sloupec `description_en`, cache `data/cache/translations.parquet`. Zkracování popisů zamítnuto.
- Na GCP poběží **jen app**, pipeline lokálně → `requirements.txt` = jen app (streamlit, pandas, pyarrow), vše ostatní v `requirements-dev.txt`.
- **Plán nasazení:** 2 Docker image – app (Cloud Run Service, malý) a pipeline (Cloud Run Job). Data verzovaná v Cloud Storage (`data/<datum>/`), pipeline po kontrole přepne `latest`, app čte `latest` → změna dat nerozbije app, rollback = přepnutí. Orchestrace GitHub Actions: `ci.yml` (testy), `deploy-app.yml`, `pipeline.yml` (ručně / cron); přihlášení Workload Identity Federation. Pořadí: nejdřív jen app s přibalenými daty.
- `title_is_latin` (podle Unicode písma, ne jazyka): 76 knih s arabským / japonským / řeckým / azbukovým názvem. Podobnost se počítá ze všech knih, ale tyto se v `similar_books` ani v app nezobrazují (předtím se objevovaly u ~1 % knih, např. Forty Rules of Love → arabské romány). Research beze změny.
- Fotky autorů z Wikipedie: `author_photo_url` (thumbnail ze stejného API, jen URL – obrázek načítá prohlížeč z Wikimedia). Fotka u 2 208 z 2 876 autorů s článkem (7 022 knih). App: portrét 3:4 se zaoblenými rohy (jako obálky) v režimu autora a v „About the author“, jinak ikonka 👤. Data se v app znovu načtou, když pipeline přepíše soubory (`data_version()`).
- Sekce autora jako u knihy: fotka vlevo (výška jako obálka), vpravo jméno, popis, bio; knihy autora ve vlastní kartě (`author_books`), nadpis podle skutečného počtu knih.

**Další krok:** ověřit hlídání nečinnosti v prohlížeči, pak Docker, Cloud Run a kill switch.
