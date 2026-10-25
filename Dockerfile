# The app only – the pipeline runs locally, its two output files are copied into the image
# (in GitHub Actions they come from Cloud Storage), so a cold start does not download anything.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements_docker.txt .
RUN pip install -r requirements_docker.txt

COPY .streamlit/ .streamlit/
COPY app/app.py app/
COPY app/static/ app/static/
COPY data/processed/books.parquet data/processed/similar_books.parquet data/processed/

# Cloud Run sends the port in $PORT
CMD streamlit run app/app.py --server.headless true --server.address 0.0.0.0 --server.port ${PORT:-8080}
