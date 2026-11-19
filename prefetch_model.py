#!/usr/bin/env python3
"""prefetch_model.py — скачать модель эмбеддингов на этапе build (Render).

Кладём в ./.embed_cache (внутри проекта) — на Render гарантированно доезжает
до runtime, в отличие от $HOME/.cache. rag_core берёт кэш из BO_EMBED_CACHE.
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

BOT_DIR = Path(__file__).resolve().parent
CACHE = os.environ.get("BO_EMBED_CACHE", str(BOT_DIR / ".embed_cache"))

REPO = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
path = snapshot_download(REPO, cache_dir=CACHE)
print("model cached:", path)
