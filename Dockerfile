FROM python:3.11-slim

WORKDIR /app

# System deps required by torch/transformers at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the model at build time so the first request isn't slow
# and the container works without internet access at runtime.
RUN python -c "from transformers import pipeline; \
    pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None)"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

