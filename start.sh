#!/usr/bin/env bash
set -e

# API (interno)
uvicorn src.api:app --host 127.0.0.1 --port 8000 &

# UI (público). Render expone el puerto en $PORT
python -m streamlit run src/app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true
