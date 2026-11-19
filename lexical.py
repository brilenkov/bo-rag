#!/usr/bin/env python3
"""lexical.py — BM25-поиск по текстам корпуса для гибрида с векторами.

Зачем: на корпусе БО чистая семантика MiniLM даёт hit@8 ≈ 0,32, BM25 —
0,39, а смесь 50/50 — 0,44 (см. work/eval_hybrid.py, 144 золотых Q/A).
Лексика ловит термины (нафс, тавакуль, ихляс), которые эмбеддер смазывает.

Стемминг: таблица «словоформа → стем» (out/lex_stems.json.gz), считается
один раз nltk Snowball (ru) и переиспользуется — на 12k параграфов сам
Snowball даёт ~240 с, что для старта бота недопустимо. Если таблицы нет,
она строится (медленно, один раз) и сохраняется; слова вне таблицы
обрезаются до 6 символов (грубый, но дешёвый fallback).

Использование:
    idx = BM25(texts)          # texts — строки в порядке коллекции
    scores = idx.scores(query) # np.array(len(texts))
    top = idx.top(query, 50)   # [(doc_idx, score), ...] по убыванию
"""
import gzip
import json
import math
import os
import re
from pathlib import Path

import numpy as np

STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за
бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей
может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз
тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один
почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об
другой хоть после над больше тот через эти нас про всего них какая много разве три эту
моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда
конечно всю между это также такой такие такое которые который которая которое свой своя
свои своих этому этим этих именно просто очень ведь разве
""".split())

WORD = re.compile(r"[а-яёa-z0-9]+", re.I)

STEM_MAP_PATH = Path(os.environ.get(
    "BO_STEM_MAP", str(Path(__file__).resolve().parent.parent / "out"
                       / "lex_stems.json.gz")))


def _default_stem_path() -> Path:
    """Ищем таблицу рядом: transcribe/out, bot/data, bot/…"""
    here = Path(__file__).resolve().parent
    for p in (here / "lex_stems.json.gz", here / "data" / "lex_stems.json.gz",
              here.parent / "out" / "lex_stems.json.gz"):
        if p.exists():
            return p
    return here.parent / "out" / "lex_stems.json.gz"


if "BO_STEM_MAP" not in os.environ:
    STEM_MAP_PATH = _default_stem_path()

_stem_map = None
_stem_fn = None
STATS = {"mode": "unknown", "unique": 0}


def _prefix(w):
    return w[:6]


def _build_map(texts) -> dict:
    """Считает стемы для всех словоформ корпуса (медленно, nltk Snowball)."""
    from nltk.stem.snowball import SnowballStemmer
    st = SnowballStemmer("russian")
    vocab = set()
    for t in texts:
        vocab.update(w for w in WORD.findall((t or "").lower()) if len(w) >= 2)
    out = {}
    for w in vocab:
        if w in STOP:
            continue
        out[w] = st.stem(w)
    return out


def ensure_stem_map(texts=None):
    """Загружает таблицу стемов; если её нет — строит из texts и сохраняет."""
    global _stem_map, _stem_fn, STATS
    if _stem_map is not None:
        return _stem_map
    try:
        if STEM_MAP_PATH.exists():
            with gzip.open(STEM_MAP_PATH, "rt", encoding="utf-8") as fh:
                _stem_map = json.load(fh)
            STATS = {"mode": "file", "unique": len(_stem_map)}
    except Exception:
        _stem_map = None
    if _stem_map is None:
        if texts is None:
            _stem_map = {}
            STATS = {"mode": "prefix", "unique": 0}
        else:
            _stem_map = _build_map(texts)
            STATS = {"mode": "built", "unique": len(_stem_map)}
            try:
                STEM_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(STEM_MAP_PATH, "wt", encoding="utf-8") as fh:
                    json.dump(_stem_map, fh, ensure_ascii=False)
            except Exception:
                pass
    _stem_fn = _stem_map.get
    return _stem_map


def stem(word: str) -> str:
    if _stem_fn is None:
        ensure_stem_map()
    return _stem_fn(word, _prefix(word))


def tokens(text: str):
    out = []
    for w in WORD.findall((text or "").lower()):
        if len(w) < 2 or w in STOP:
            continue
        out.append(stem(w))
    return out


class BM25:
    """BM25 (k1=1.2, b=0.75) по готовому списку текстов."""

    def __init__(self, texts, k1: float = 1.2, b: float = 0.75):
        ensure_stem_map(texts)
        self.k1, self.b = k1, b
        self.N = len(texts)
        self.lens = np.zeros(self.N, dtype=np.float32)
        post = {}  # терм -> (список doc_id, список tf)
        for i, t in enumerate(texts):
            ts = tokens(t)
            self.lens[i] = len(ts)
            counts = {}
            for w in ts:
                counts[w] = counts.get(w, 0) + 1
            for w, c in counts.items():
                if w in post:
                    post[w][0].append(i)
                    post[w][1].append(c)
                else:
                    post[w] = ([i], [c])
        self.avg = float(self.lens.mean()) or 1.0
        self.inv = {}
        for w, (ids, tfs) in post.items():
            if len(ids) < 2:  # hapax не различает документы
                continue
            n = len(ids)
            idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            self.inv[w] = (np.asarray(ids, dtype=np.int32),
                           np.asarray(tfs, dtype=np.float32), idf)
        self.vocab = set(self.inv)
        self._denom = self.k1 * (1 - self.b + self.b * self.lens / self.avg)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(tokens(query)):
            item = self.inv.get(t)
            if not item:
                continue
            ids, tfs, idf = item
            s[ids] += idf * tfs * (self.k1 + 1) / (tfs + self._denom[ids])
        return s

    def top(self, query: str, k: int = 50):
        s = self.scores(query)
        if not len(s):
            return []
        k = min(k, len(s))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]


# ---------------- сериализация индекса (быстрый старт) ----------------

def save_index(path, bm: BM25, ids) -> None:
    """Сохраняет BM25 в npz: словарь терминов, postings (CSR-подобно), idf."""
    terms = list(bm.inv)
    vocab_index = {t: i for i, t in enumerate(terms)}
    counts = np.array([len(bm.inv[t][0]) for t in terms], dtype=np.int64)
    offsets = np.zeros(len(terms) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    doc_ids = np.concatenate([bm.inv[t][0] for t in terms]) if terms else \
        np.zeros(0, dtype=np.int32)
    tfs = np.concatenate([bm.inv[t][1] for t in terms]) if terms else \
        np.zeros(0, dtype=np.float32)
    idf = np.array([bm.inv[t][2] for t in terms], dtype=np.float32)
    np.savez_compressed(
        path, terms=np.array(terms, dtype=object) if terms else np.zeros(0, object),
        counts=counts, offsets=offsets, doc_ids=doc_ids, tfs=tfs, idf=idf,
        lens=bm.lens, k1=np.float32(bm.k1), b=np.float32(bm.b),
        avg=np.float32(bm.avg), ids=np.array([str(i) for i in ids]))


class BM25Loaded(BM25):
    """BM25 из npz (без перетокенизации корпуса)."""

    def __init__(self, d):
        self.k1 = float(d["k1"])
        self.b = float(d["b"])
        self.lens = d["lens"]
        self.avg = float(d["avg"])
        self.N = int(len(self.lens))
        terms = list(d["terms"])
        offs = d["offsets"]
        doc_ids = d["doc_ids"]
        tfs = d["tfs"]
        idf = d["idf"]
        self.inv = {}
        for i, t in enumerate(terms):
            a, b = int(offs[i]), int(offs[i + 1])
            self.inv[str(t)] = (doc_ids[a:b], tfs[a:b], float(idf[i]))
        self.vocab = set(self.inv)
        self._denom = self.k1 * (1 - self.b + self.b * self.lens / self.avg)


def load_index(path):
    """(ids, BM25Loaded) или (None, None), если файла нет."""
    try:
        with np.load(path, allow_pickle=True) as d:
            ids = [str(x) for x in d["ids"]]
            return ids, BM25Loaded(d)
    except Exception:
        return None, None
