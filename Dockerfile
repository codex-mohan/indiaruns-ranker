FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so first sandbox run doesn't wait for HuggingFace Hub
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached.')"

# Copy source and demo assets
COPY src/ src/
COPY sandbox/ sandbox/
COPY data/ data/

# Default: launch the Gradio sandbox. The sandbox auto-precomputes matching
# artifacts for the bundled sample or uploaded JSONL, then ranks.
CMD ["python", "sandbox/app.py"]
