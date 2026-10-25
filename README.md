# Book Recommender

Enter a book, get similar books — based on what other users read and rate.

> 🚧 Work in progress

## How it works

1. **Pipeline** – download goodbooks-10k, clean it, add covers, compute similar books (item-item)
2. **Research** – compare popularity, item-item, ALS and a two-tower neural network; item-item wins for "similar books"
3. **Serving** – app on GCP that scales to zero, so it costs money only when a request comes in

## Project structure

Folders by where the code runs – ☁️ cloud (Cloud Run), 💻 local, 🔧 build and deployment, 📄 docs and settings:

```
book_recommender/
│
├── app/                          ☁️  goes into the Docker image and runs on Cloud Run
│   ├── app.py                        Streamlit app – pick a book or an author, see similar books
│   ├── static/                       header photo, page shown after the idle timeout
│   └── test_app.py                   smoke test with the real data – run on GitHub before deploying, not in the image
│
├── pipeline/                     💻 makes the data the app reads
│   ├── pipeline.py                   raw data -> books, interactions, similar books
│   └── test_pipeline.py              tests of the pipeline
│
├── research/                     💻 experiments, not used by the app
│   ├── eda.ipynb                     data exploration
│   ├── experiments.ipynb             model comparison: split, features, models, training, evaluation, Excel report
│   └── output/                       results, splits, features, models – not in git
│
├── data/                         📦 raw/, cache/, processed/ – made by the pipeline, not in git;
│                                    the two app files in processed/ go via Cloud Storage into the image
│
├── requirements_docker.txt       ☁️  libraries of the app (in the image)
├── requirements_local.txt        💻 everything for local work – pipeline, research, tests (includes the above)
│
├── Dockerfile, .dockerignore     🔧 how the image is packed (what goes to the cloud)
├── .github/workflows/deploy.yml  🔧 how it is deployed (GitHub Actions)
├── .streamlit/config.toml        ☁️ 💻 Streamlit settings, used in the cloud and locally
│
└── README.md, pytest.ini, .gitignore   📄
```

Only `app/` (without the test), `.streamlit/`, the libraries from `requirements_docker.txt` and the two data files go
to the cloud – everything else stays local or on GitHub. `.dockerignore` enforces it: everything is excluded except
these files, so nothing else can slip into the image by mistake.

## Data sources

| Source | What we take | Why |
|---|---|---|
| [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) | 6M ratings from 53k users on 10k books, book metadata | The model needs user–book interactions. Goodreads closed its API in 2020; this is a public snapshot of it. |
| [Open Library](https://openlibrary.org/developers/api) | Book covers | Larger and more complete covers than goodbooks-10k. Open Library has no per-user ratings, so it is used only to enrich books. |
| [Wikipedia](https://en.wikipedia.org/api/rest_v1/) | Short author info (description + first paragraph) and author photo (URL only) | Real facts instead of generated text; found for 2,876 of 3,888 authors. Text under CC BY-SA, the app links the source. |
| [Unsplash](https://unsplash.com) | Header photo by [Eldem Ozdemir](https://unsplash.com/@eldemozd) | Free under the Unsplash License; credited in the app's About section. |

goodbooks-10k by [zygmuntz](https://github.com/zygmuntz) ([FastML](http://fastml.com/goodbooks-10k)) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Used: `ratings.csv`, `books.csv`, reader tags (`book_tags.csv`, `tags.csv`) for genres and `books_xml.zip`
(Goodreads XML files from the same repository) for book descriptions.

## Pipeline – `pipeline/pipeline.py`

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

**3. Similar books** – item-item similarity (see below) from all ratings ≥ 4 → the 50 most similar books for every book
→ `similar_books.parquet` (`book_id`, `rank`, `similar_book_id`, `score`; 10,000 × 20 rows).
The answers are computed once here, so the app only looks them up instead of crunching 6M ratings on every request.
The similarity uses all books, but the 76 books with titles in Arabic, Japanese, Greek or Cyrillic script (`title_is_latin`)
are neither listed nor recommended – the app is English.

```
interactions ──(item-item)──► similar_books
app: "The Hobbit" → books (find book_id) → similar_books (50 similar) → books (titles, authors, covers)
```

| Output in `data/processed/` | Content | Used by |
|---|---|---|
| `books.parquet` | 10k books – title, author, ISBN, cover | app, research |
| `interactions.parquet` | 6M ratings – who rated which book | step 3, research (not the app) |
| `similar_books.parquet` | 50 most similar books for every book | app |

### How genres are built

goodbooks-10k has no genre column – only 34,252 free-form tags that readers put on books (`book_tags.csv`, up to 100 tags per book with vote counts).
Most of them are not genres, so they are turned into genres in three steps (`GENRES`, `tag_genres`, `pick_genres` in `pipeline/pipeline.py`):

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
pip install -r requirements_local.txt   # pipeline, research and tests (runs locally)
python pipeline/pipeline.py             # ingest -> etl -> similar books (first run ~1.5 h: covers, author info, translations – all cached)
streamlit run app/app.py                # app on http://localhost:8501 (from the repo root)
pytest
jupyter notebook research/experiments.ipynb
```

`requirements_docker.txt` holds only what the app needs (Streamlit, pandas, pyarrow) – that is what gets deployed.

## Deployment and cost safety

The app is a **Streamlit** app (`app/app.py`), to be deployed on **Cloud Run**, reading only `books.parquet` and `similar_books.parquet`.
The data is static, so only the app runs on GCP – the pipeline runs locally and its two output files are shipped with the app.
Cloud Run scales to zero, but Streamlit keeps a connection open while the page is open and Cloud Run bills that time,
so the deployment gets several safety limits:

| Safety limit | Protects against | Reacts |
|---|---|---|
| `--max-instances=1` | traffic spikes, bots, attacks | immediately – never more than one instance |
| `--timeout=300` | endless connections | after 5 min |
| idle check in the page (JavaScript) | forgotten open tabs | after 2 min without activity the page closes the connection |
| spend cap 100 Kč (≈ 4 USD) per month on Cloud Run in this project, e-mails at 50 / 80 / 100 % | slow cost leaks, e.g. a bot holding the page for days | Google pauses the Cloud Run service (5xx) until the next month or until the cap is lifted; enforcement lags a few hours, overage is billed |

Worst case ≈ 5 USD per month (cap + a few hours of enforcement delay); normal portfolio traffic fits the Cloud Run free tier.
The spend cap replaced the planned kill switch (budget → Pub/Sub → function), which is no longer needed.

## Deployment on GCP

Steps are filled in as they are done; every command below was run and checked.

**How it fits together**

```
 LOCAL                           GITHUB                                  GOOGLE CLOUD (project book-recommender-512711)
 ─────                           ──────                                  ──────────────────────────────────────────────

 code (app/app.py…) ── git push ──►  repo martincho48/book_recommender
                                         │
                                         ▼
                                    GitHub Actions  ──── 1. "I am this repo, branch main" ─────►  Workload Identity Federation
                                    (deploy.yml)    ◄─── short-lived token (1 h) ──────────────  (the gatekeeper)
                                         │
                                         │  2. get the data ◄────────────────────────────────────  Cloud Storage (bucket)
                                         │                                                         books + similar_books
                                         │  3. test the app (on GitHub)
                                         │  4. build the Docker image (app + data in one box)
                                         │  5. push the image ───────────────────────────────────►  Artifact Registry
                                         │                                                         (keeps the 2 newest)
                                         │  6. "run this image" ─────────────────────────────────►  Cloud Run (service book-recommender)
                                                                                                      runs as an account with no roles
 pipeline.py ── data ── gcloud storage cp (only when the data changes) ──────────────────────────►  Cloud Storage

                                                                                                    spend cap 100 Kč ── guards Cloud Run

 VISITOR:  browser ── https://book-recommender-….run.app ──►  Cloud Run  (starts the image if it is asleep)
```

| Part | Think of it as | What it does |
|---|---|---|
| **Docker image** | a packed box | the app, Python, libraries and data in one package – runs the same everywhere |
| **GitHub Actions** | a worker on GitHub | after a push runs `deploy.yml`: tests the app, packs the box, sends it to the cloud |
| **Workload Identity Federation** | the gatekeeper | lets the worker into the project only when it comes from this repository and `main` |
| **`github-deployer`** | the worker's badge | what the worker may do: push the image, read the data, run the app – nothing else |
| **Cloud Storage** (bucket) | the data store | the two data files; packed into the box at build time |
| **Artifact Registry** | the box store | built images, the 2 newest are kept, older ones are deleted automatically |
| **Cloud Run** | the counter | starts the box when a visitor comes, puts it to sleep when nobody does |
| **`book-recommender-run`** | the app's identity | the account the app runs as – no roles at all |
| **Spend cap** | the fuse | pauses Cloud Run at 100 Kč a month |

When a visitor opens the page, only Cloud Run and the image in Artifact Registry are involved – GitHub and the bucket
are used only while deploying. Where to look: GitHub **Actions** tab (every deployment and its steps); GCP console
**Cloud Run** (URL, logs, revisions), **Artifact Registry**, **Cloud Storage → Buckets**, **IAM → Service accounts**,
**Billing → Budgets & caps**.

- **Project** `book-recommender-512711`, **region** `europe-west1` (Belgium – close to Central Europe, lowest price tier).
- **Data** – the pipeline runs locally; its two output files go to a Cloud Storage bucket once and are copied
  into the image at build time, so a cold start downloads nothing.
- **Login from GitHub** – Workload Identity Federation: no service-account keys in GitHub, and only this repository's
  `main` branch may deploy.
- **Files** – `Dockerfile` (python:3.10-slim, app + the two data files only), `.dockerignore` (allowlist),
  `.streamlit/config.toml` (static serving for the idle page, no usage stats), `requirements_docker.txt` (pinned versions).

| # | Step | Status |
|---|---|---|
| 1 | Project, billing account linked | ✅ done |
| 2 | Docker image files in the repo | ✅ done |
| 3 | Enable APIs: Cloud Run, Artifact Registry, Cloud Storage, IAM Credentials | ✅ done |
| 4 | Artifact Registry repository for the images, cleanup policy (keep the 2 newest) | ✅ done |
| 5 | Cloud Storage bucket with `books.parquet` and `similar_books.parquet` | ✅ done |
| 6 | Service accounts (deployer, app) + Workload Identity Federation for GitHub | ✅ done |
| 7 | GitHub Actions workflow `.github/workflows/deploy.yml` (app test → image → Cloud Run) | ⏳ written, not run yet |
| 8 | Spend cap 100 Kč/month on Cloud Run (Billing → Budgets & caps → Spend cap enforcement; console only) | ✅ done |

**One-time setup** (Git Bash, `gcloud auth login` first)

```bash
# 3 – APIs (Cloud Storage is on by default in a new project)
gcloud config set project book-recommender-512711
gcloud services enable run.googleapis.com artifactregistry.googleapis.com iamcredentials.googleapis.com

# 4 – Artifact Registry repository for the Docker images
gcloud artifacts repositories create book-recommender --repository-format=docker --location=europe-west1 --description="Images of the Book Recommender app"
# keep only the 2 newest images (current + one to roll back to); Google runs the cleanup about once a day
cat > cleanup-policy.json <<'EOF'
[
  {"name": "keep-latest-2", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any"}}
]
EOF
gcloud artifacts repositories set-cleanup-policies book-recommender --location=europe-west1 --policy=cleanup-policy.json --no-dry-run

# 5 – private bucket with the app data (re-run the cp after the pipeline changes the data)
gcloud storage buckets create gs://book-recommender-512711-data --location=europe-west1 --uniform-bucket-level-access --public-access-prevention
gcloud storage cp data/processed/books.parquet data/processed/similar_books.parquet gs://book-recommender-512711-data/

# 6 – service accounts with the smallest roles, and the federation (only this repo, branch main)
D=github-deployer@book-recommender-512711.iam.gserviceaccount.com
R=book-recommender-run@book-recommender-512711.iam.gserviceaccount.com
gcloud iam service-accounts create book-recommender-run --display-name="Book Recommender app (Cloud Run)"
gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer"
gcloud projects add-iam-policy-binding book-recommender-512711 --member="serviceAccount:$D" --role="roles/run.admin" --condition=None
gcloud artifacts repositories add-iam-policy-binding book-recommender --location=europe-west1 --member="serviceAccount:$D" --role="roles/artifactregistry.writer"
gcloud storage buckets add-iam-policy-binding gs://book-recommender-512711-data --member="serviceAccount:$D" --role="roles/storage.objectViewer"
gcloud iam service-accounts add-iam-policy-binding $R --member="serviceAccount:$D" --role="roles/iam.serviceAccountUser"
gcloud iam workload-identity-pools create github --location=global --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc github-repo --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='martincho48/book_recommender' && assertion.ref=='refs/heads/main'"
gcloud iam service-accounts add-iam-policy-binding $D --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/497923768669/locations/global/workloadIdentityPools/github/attribute.repository/martincho48/book_recommender"
```

### Updating the deployed app (once step 7 is done)

| Change | What to do |
|---|---|
| App code – `app/app.py`, `app/static/`, `requirements_docker.txt`, `Dockerfile`, `.streamlit/` | commit + `git push` to `main` – deployed automatically |
| Data – pipeline re-run | upload the two files again (`gcloud storage cp`, step 5), then GitHub → **Actions** → **Deploy** → **Run workflow** |
| README, `research/`, `pipeline/pipeline.py` | nothing is deployed – they do not change the app |
| A push to another branch | nothing is deployed |

The workflow runs the tests, takes the data from the bucket, builds the image, pushes it to Artifact Registry and
deploys it to Cloud Run (a few minutes; progress in the **Actions** tab). If the tests or the build fail, the previous
version keeps running. A bad version can be rolled back in the console: **Cloud Run → book-recommender → Revisions**.

### Who can deploy

Only the repository owner:

- only users with write access can push to `main` or start **Run workflow** – nobody else is a collaborator;
- a public repository can be read and forked, not written to; a fork or a pull request cannot deploy,
  because the federation accepts only `martincho48/book_recommender` on `refs/heads/main` – a pull request is
  deployed only after the owner merges it;
- the GCP project has no other members.

So everything rests on the GitHub and Google accounts – both with two-factor authentication.

### Accounts used by the deployment

| Account | Used by | Allowed to |
|---|---|---|
| `github-deployer@book-recommender-512711.iam.gserviceaccount.com` | GitHub Actions (via Workload Identity Federation, no key) | deploy to Cloud Run, push images to the `book-recommender` repository, read the data bucket, run the app as `book-recommender-run` |
| `book-recommender-run@book-recommender-512711.iam.gserviceaccount.com` | the running app | nothing – the app needs no GCP access (data is in the image, covers come from the internet) |

Service accounts are accounts for programs, not people: no password, no login, only the roles listed above.
Without its own account the app would run as the Compute Engine default account, which can edit the whole project.

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
