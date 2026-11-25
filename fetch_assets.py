#!/usr/bin/env python3
"""fetch_assets.py — скачивает плоский стор + GGUF-модель из release-ассета.

Ассет bo_rag_bot_assets_v13.tar.gz (приватный репо → нужен env GH_PAT):
    data/vectors_par.npy, data/vectors_qa.npy          — вектора (384d)
    data/payloads.db, data/meta.json                  — тексты и метаданные (sqlite)
    data/lex_par.npz, data/lex_qa.npz                 — BM25-индекс (гибрид)
    data/lex_stems.json.gz                            — стемы для BM25
    models/…-Q4_K_M.gguf                               — эмбеддер llama.cpp

Идемпотентно: если data/meta.json и models/*.gguf на месте — выходит.
Запускается на build (Render buildCommand), артефакты проекта
доживают до runtime.
"""
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent
ASSET_URL = ("https://api.github.com/repos/brilenkov/brain-tools/"
             "releases/assets/792816672")
# llama-cpp-python на PyPI только sdist (компиляция падает на Render),
# поэтому wheel собран локально (glibc 2.36) и лежит рядом ассетом.
WHEEL_URL = ("https://api.github.com/repos/brilenkov/brain-tools/"
             "releases/assets/742154872")
WHEEL_NAME = "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl"  # внутри — portable-сборка (GGML_NATIVE=OFF), имя стандартное
# карта «дата сохбета → message_id в группе-источнике» для ссылок в «Источниках»
LINKS_URL = ("https://api.github.com/repos/brilenkov/brain-tools/"
             "releases/assets/743517149")


def already_ok() -> bool:
    return ((BOT_DIR / "data" / "meta.json").exists()
            and list((BOT_DIR / "models").glob("*.gguf")) != []
            and (BOT_DIR / "data" / "sohbet_links.json").exists()
            and (BOT_DIR / "data" / "lex_par.npz").exists()
            and (BOT_DIR / "data" / "lex_qa.npz").exists()
            and (BOT_DIR / "data" / "lex_stems.json.gz").exists()
            and (BOT_DIR / WHEEL_NAME).exists())


def download(url: str, pat: str, dest) -> None:
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {pat}",
        "Accept": "application/octet-stream",
        "User-Agent": "bo-rag-bot-build",
    })
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)


def main() -> int:
    if already_ok():
        print("assets уже на месте — пропускаю скачивание")
        return 0
    pat = os.environ.get("GH_PAT", "").strip()
    if not pat:
        sys.exit("Нужен env GH_PAT (репо приватный)")

    req = urllib.request.Request(ASSET_URL, headers={
        "Authorization": f"token {pat}",
        "Accept": "application/octet-stream",
        "User-Agent": "bo-rag-bot-build",
    })
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        print("скачиваю bo_rag_bot_assets_v13.tar.gz …", flush=True)
        download(ASSET_URL, pat, tmp.name)
        tmp_path = tmp.name
    print("скачано, распаковываю …", flush=True)
    with tarfile.open(tmp_path) as tar:
        tar.extractall(BOT_DIR, filter="data")
    os.unlink(tmp_path)

    print("скачиваю wheel llama-cpp-python …", flush=True)
    download(WHEEL_URL, pat, BOT_DIR / WHEEL_NAME)

    print("скачиваю sohbet_links.json …", flush=True)
    download(LINKS_URL, pat, BOT_DIR / "data" / "sohbet_links.json")

    if not already_ok():
        sys.exit("после распаковки assets не найдены — битый архив?")
    print("assets готовы:", sorted(p.name for p in
                                 (BOT_DIR / "data").iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
