FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

# libgomp1: OpenMP runtime required by faiss-cpu
# tesseract-ocr (+ English data) and poppler-utils (pdftoppm): OCR for scanned PDFs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# The build context is ./backend; it is copied as the `backend` package.
COPY . /app/backend

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
