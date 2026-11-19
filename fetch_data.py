#!/usr/bin/env python3
"""fetch_data.py — скачать снапшот Qdrant-базы из GitHub Release (build step).

Ассет приватного репо → нужен GH_PAT (env, задаётся в Render Dashboard).
Идемпотентно: если data/qdrant_data уже есть — пропускает.
"""
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent
TARGET = BOT_DIR / "data" / "qdrant_data"
ASSET_URL = ("https://api.github.com/repos/brilenkov/brain-tools/"
             "releases/assets/740864217")


def main():
    if TARGET.exists():
        print("data/qdrant_data уже есть — пропуск")
        return
    pat = os.environ.get("GH_PAT", "").strip()
    if not pat:
        sys.exit("Нужен env GH_PAT для скачивания data snapshot "
                 "(приватный репо)")
    req = urllib.request.Request(
        ASSET_URL,
        headers={"Authorization": f"token {pat}",
                 "Accept": "application/octet-stream",
                 "User-Agent": "bo-rag-bot-build"})
    with urllib.request.urlopen(req, timeout=600) as r, \
            tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        tmp.write(r.read())
        tmp.flush()
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp.name) as tar:
            tar.extractall(TARGET.parent, filter="data")
    print("data snapshot распакован:", TARGET)


if __name__ == "__main__":
    main()
