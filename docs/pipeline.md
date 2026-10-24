# Datová pipeline

## Pořadí spuštění

```
python src/ingest.py   # 1. stáhne surová CSV
python src/etl.py      # 2. vyčistí → books.parquet + interactions.parquet
python src/enrich.py   # 3. přidá obálky z Open Library do books.parquet
```

Pořadí je důležité: `etl.py` při každém běhu přepíše `books.parquet` bez obálek → po něm musí vždy běžet `enrich.py` (díky cache trvá pár sekund).

## Soubory

```
data/
├── raw/                      ← ingest.py
│   ├── books.csv, ratings.csv           (používané)
│   └── to_read.csv, tags.csv, book_tags.csv  (zatím nepoužívané)
└── processed/
    ├── books.parquet         ← etl.py, doplní enrich.py
    ├── interactions.parquet  ← etl.py
    └── ol_covers.parquet     ← enrich.py (cache isbn → ol_cover_id)
```

`data/` je v `.gitignore` – necommituje se, na jiném stroji se vytvoří spuštěním pipeline.

## 1. `ingest.py` – stažení dat

- Stáhne 5 CSV z [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) (~100 MB) do `data/raw/`.
- Existující soubory přeskočí, `--force` stáhne znovu.
- Stahuje přes dočasný `.part` soubor → po pádu nezůstane rozbitý soubor.

## 2. `etl.py` – čištění

`ratings` a `books` se **neslučují** do jedné tabulky – zůstávají oddělené, propojené přes `book_id`
(fakta + dimenze). Model trénuje jen z `interactions`, `books` slouží k zobrazení a hledání.

**books** (10 000 řádků, žádný se nemaže – každá kniha má hodnocení):
- z 23 sloupců ponechá 10
- `isbn` doplní nulami zleva na 10 znaků
- `isbn13` **přepočítá z `isbn`** – ve zdroji je uložené na 12 platných číslic, chybí kontrolní číslice
- chybějící `original_title` doplní z `title`, rok jako celé číslo
- jazyky `eng`, `en-US`, `en-GB`, `en-CA` → `en`
- zástupný obrázek „nophoto“ → NaN

**interactions** (z `ratings`, 5 976 479 řádků):
- maže NaN, duplicitní dvojice (user, book), hodnocení mimo 1–5, hodnocení neexistujících knih
  (v datech zatím nic z toho není – pojistka)

**Validace** (`assert`, při chybě se nic neuloží): unikátní `book_id`, žádné NaN ani duplicity,
hodnocení 1–5, vazba oběma směry (každé hodnocení má knihu, každá kniha má hodnocení).

## 3. `enrich.py` – obálky z Open Library

Dvě části v jednom skriptu:

1. **Stažení přes API** – pro ISBN, která ještě nejsou v cache, zjistí `cover_i`
   dávkovým hledáním `search.json?q=isbn:(a OR b …)` po 50 ISBN (~190 dotazů při prvním běhu)
   a nové výsledky připíše do `ol_covers.parquet`.
2. **Spárování** – připojí cache k `books.parquet` přes `isbn` a přidá sloupce:
   - `ol_cover_id` – ID obálky v Open Library (NaN, pokud se nenašla)
   - `cover_url` – obálka z Open Library → záloha Goodreads `image_url` → NaN

Chování při dalším spuštění:
- knihy, které už v cache jsou, se nestahují znovu
- nenalezené knihy se zkouší znovu při každém běhu (Open Library mohla obálku mezitím přidat)
- nové knihy se dotáhnou automaticky
- úplně znovu: smazat `ol_covers.parquet`

Proč přes `cover_i`: URL `covers.openlibrary.org/b/id/<cover_i>-M.jpg` nemá rate limit,
URL přes ISBN má limit 100 požadavků / 5 min na IP. Obrázky se nestahují – aplikace jen vloží URL
a obrázek načte prohlížeč uživatele.

Volitelně: proměnná `OPENLIBRARY_CONTACT` (např. e-mail) → Open Library povolí 3 req/s místo 1.

## Proměnné prostředí

| Skript | Proměnná | Výchozí |
|---|---|---|
| `ingest.py` | `OUTPUT_PATH` | `data/raw` |
| `etl.py` | `INPUT_PATH`, `OUTPUT_PATH` | `data/raw`, `data/processed` |
| `enrich.py` | `DATA_PATH`, `OPENLIBRARY_CONTACT` | `data/processed`, – |

Na GCP se cesty přesměrují do Cloud Storage, kód zůstává stejný.
