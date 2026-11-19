#!/usr/bin/env python3
"""rag_core.py — ядро RAG для цифрового двойника БО (версия для Render free tier).

Отличия от qdrant-версии (scripts/rag_core.py в репо transcribe):
  - поиск — чистый numpy по плоскому стору (vectors_*.npy + payloads_*.json.gz):
    Qdrant local держит весь индекс в RAM (~230 МБ anon) — не влезает в 512 МБ;
  - эмбеддер — llama.cpp GGUF Q8_0 той же модели MiniLM-L12 (~180 МБ anon,
    веса mmap-ятся из файла и вытесняются ядром); fastembed/onnxruntime
    материализует fp32-копию весов в anon-память (~430 МБ) — OOM на Render;
  - LLM-вызов — in-process urllib (без subprocess + второго интерпретатора).

Совместимость: cos(gguf_q8, fastembed_fp16) = 0.99-0.998 на контрольных
запросах, топ-k совпадает — плоский стор экспортирован из qdrant-базы,
собранной fastembed-версией (см. transcribe/scripts/export_flat_store.py).

Поток: вопрос → (HyDE) → embed → numpy cosine top-k (параграфы + QA)
     → промпт режима (zhivoy|publication) → LLM через Go API → ответ + цитаты.

Приватность: корпус уходит ТОЛЬКО в zero-retention модели Go API
(minimax-m3 по умолчанию; fallback deepseek-v4.1-flash).
"""
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BO_DATA_DIR", str(BOT_DIR / "data")))
MODEL_PATH = Path(os.environ.get(
    "BO_GGUF_PATH",
    str(BOT_DIR / "models" /
        "paraphrase-multilingual-MiniLM-L12-118M-v2-Q4_K_M.gguf")))

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2 (GGUF Q8_0, llama.cpp)"
EMBED_DIM = 384

# Модели генерации (zero-retention, разрешены для корпуса)
GEN_MODEL = os.environ.get("BO_GEN_MODEL", "minimax-m3")
GEN_MODEL_FALLBACK = "deepseek-v4.1-flash"

# Порог "не знаю": если лучший результат хуже — честно говорим, что не говорил
MIN_SCORE = float(os.environ.get("BO_MIN_SCORE", "0.60"))

# Порог для показа в «Источниках» (fallback-режим, когда LLM не пометил,
# какие фрагменты использовал): чуть строже MIN_SCORE, чтобы не показывать
# отдалённые по теме фрагменты
CITE_MIN_SCORE = float(os.environ.get("BO_CITE_MIN_SCORE", "0.63"))

# Гибридный поиск: score = w*cosine + (1-w)*BM25. По eval 50/50 заметно лучше
# чистого cosine на редких терминах (нафс, нисбат и т.п.).
LEX_WEIGHT = float(os.environ.get("BO_LEX_WEIGHT", "0.5"))
# HyDE по умолчанию выключен: по eval он ухудшает hit@8 и добавляет
# лишний вызов LLM (~5-10 с). Включается BO_HYDE=1.
HYDE_ENABLED = os.environ.get("BO_HYDE") == "1"

log = logging.getLogger("rag_core")


def mem_mb() -> int:
    """Anon RSS процесса в МБ (то, за что реально убивает OOM-killer)."""
    try:
        s = open("/proc/self/status").read()
        return int(s.split("RssAnon:")[1].split()[0]) // 1024
    except Exception:
        return -1


# ---------------- эмбеддер (llama.cpp GGUF) ----------------

_lock = threading.Lock()
_embed_lock = threading.Lock()
_embedder = None
_store = None


def get_embedder():
    """Ленивый синглтон Llama(embedding=True). ~180 МБ anon после загрузки."""
    global _embedder
    with _lock:
        if _embedder is None:
            from llama_cpp import Llama
            t0 = time.time()
            _embedder = Llama(model_path=str(MODEL_PATH), embedding=True,
                              n_ctx=512, n_batch=16, n_ubatch=16,
                              n_threads=1, use_mmap=True, use_mlock=False,
                              verbose=False)
            log.info("embedder loaded in %.1fs, anon=%dMB",
                     time.time() - t0, mem_mb())
    return _embedder


def embed_text(text: str):
    import numpy as np
    llm = get_embedder()
    # llama.cpp не потокобезопасен: веб-чат и Telegram могут прийти одновременно
    with _embed_lock:
        v = np.array(llm.embed(text), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------- плоский стор (numpy cosine) ----------------

class FlatStore:
    """Точный cosine-поиск по всем векторам (10613 × 384 — миллисекунды,
    ANN-индекс не нужен). Вектора: 16 МБ в RAM; payload-ы — в sqlite
    (6 МБ, читаются лениво построчно — экономим ~50 МБ anon против json)."""

    def __init__(self, data_dir: Path):
        import numpy as np
        import sqlite3
        t0 = time.time()
        self.tables = {}
        self._db_lock = threading.Lock()
        self.db = sqlite3.connect(f"file:{data_dir / 'payloads.db'}?mode=ro",
                                  uri=True, check_same_thread=False)
        for tag in ("par", "qa"):
            V = np.load(data_dir / f"vectors_{tag}.npy").astype(
                np.float32, copy=False)
            V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            self.tables[tag] = V
        # внешние книги (Юнг, Абрамова и т.п.) — только reference, по умолчанию
        # исключаем из выдачи: бот отвечает по материалам БО.
        self.blocked = {}
        for tag in ("par", "qa"):
            mask = np.zeros(self.tables[tag].shape[0], dtype=bool)
            try:
                rows = self.db.execute(
                    f"SELECT id FROM {tag} WHERE "
                    "json_extract(payload, '$.source_type') = 'reference'"
                ).fetchall()
            except Exception:
                rows = self.db.execute(
                    f"SELECT id FROM {tag} WHERE "
                    "payload LIKE '%\"source_type\": \"reference\"%'"
                ).fetchall()
            for (i,) in rows:
                if 0 <= i < mask.shape[0]:
                    mask[i] = True
            self.blocked[tag] = mask
            log.info("store %s: blocked reference=%d/%d", tag,
                     int(mask.sum()), mask.shape[0])
        # предсобранный BM25 (lex_{tag}.npz) — без него работаем на cosine
        self._lex = {}
        for tag in ("par", "qa"):
            idx = None
            try:
                import lexical
                ids, idx = lexical.load_index(data_dir / f"lex_{tag}.npz")
                if idx is None or ids is None or \
                        len(ids) != self.tables[tag].shape[0]:
                    idx = None
            except Exception as e:
                log.warning("lex %s load failed: %s", tag, e)
            self._lex[tag] = idx
            log.info("lex %s: %s", tag,
                     f"terms={len(idx.inv)}" if idx is not None else "off")
        log.info("store loaded in %.1fs, anon=%dMB", time.time() - t0, mem_mb())

    def _payloads(self, tag: str, ids) -> list:
        marks = ",".join(str(int(i)) for i in ids)
        with self._db_lock:
            cur = self.db.execute(
                f"SELECT id, payload FROM {tag} WHERE id IN ({marks})")
            got = {i: json.loads(p) for i, p in cur.fetchall()}
        return [got[int(i)] for i in ids]

    def search(self, tag: str, vec, k: int, question: str = None):
        """Возвращает (hits, best_cosine). Сортировка — по гибридному score,
        в поле score остаётся чистый cosine (для цитат и порога)."""
        import numpy as np
        V = self.tables[tag]
        sims = V @ vec
        blocked = self.blocked.get(tag)
        if blocked is not None and blocked.any():
            best_cos = float(sims[~blocked].max()) if (~blocked).any() else 0.0
        else:
            best_cos = float(sims.max()) if len(sims) else 0.0
        fused = sims
        lex = self._lex.get(tag)
        if question and lex is not None:
            try:
                ls = lex.scores(question)
                mx = float(ls.max())
                if mx > 0:
                    fused = (LEX_WEIGHT * sims / (best_cos or 1.0)
                             + (1 - LEX_WEIGHT) * ls / mx)
            except Exception as e:
                log.warning("lex search failed: %s", e)
        if blocked is not None and blocked.any():
            if fused is sims:
                fused = sims.copy()
            else:
                fused = fused.copy()
            fused[blocked] = -1e9
        k = min(k, V.shape[0])
        idx = np.argpartition(-fused, k - 1)[:k]
        idx = idx[np.argsort(-fused[idx])]
        return ([{"score": round(float(sims[i]), 4),
                  "rank": round(float(fused[i]), 4), **p}
                 for i, p in zip(idx, self._payloads(tag, idx))], best_cos)

    def counts(self):
        return {tag: V.shape[0] for tag, V in self.tables.items()}


def get_store() -> FlatStore:
    global _store
    with _lock:
        if _store is None:
            _store = FlatStore(DATA_DIR)
    return _store


# ---------------- LLM через Go API (in-process urllib) ----------------

def call_llm(prompt: str, system: str = "", max_tokens: int = 4000,
             model: str = None) -> str:
    """In-process вызов go_llm_call.call_model (без subprocess — экономия
    ~80 МБ пиковой памяти). RATE_LIMITED → fallback-модель."""
    import go_llm_call
    model = model or GEN_MODEL
    try:
        return go_llm_call.call_model(model, prompt, system=system,
                                      max_tokens=max_tokens)
    except go_llm_call.RateLimited:
        if model != GEN_MODEL_FALLBACK:
            log.warning("RATE_LIMITED на %s — fallback %s",
                        model, GEN_MODEL_FALLBACK)
            return go_llm_call.call_model(GEN_MODEL_FALLBACK, prompt,
                                          system=system, max_tokens=max_tokens)
        raise RuntimeError("RATE_LIMITED на обеих моделях")


def hyde(question: str) -> str:
    """Гипотетический ответ БО для улучшения поиска (1 короткий LLM-вызов)."""
    prompt = (
        "Напиши короткий (4-6 предложений) гипотетический ответ суфийского "
        "шейха Бориса Олеговича на вопрос ученика. Это нужно только для "
        "семантического поиска, не для показа пользователю. Используй лексику "
        "накшбандийского тариката (ихляс, тарикат, мюрид, сохбет, дхикр, "
        "тавакуль, нафс, хидмат). Не выдумывай фактов.\n\n"
        f"Вопрос ученика: {question}\n\nГипотетический ответ:"
    )
    try:
        return call_llm(prompt, max_tokens=600)
    except Exception:
        return question  # fallback: ищем по самому вопросу


def retrieve(question: str, k_par: int = 8, k_qa: int = 6,
             use_hyde: bool = None) -> dict:
    """Гибридный поиск (cosine + BM25). Возвращает paragraphs/qa + best_cos."""
    if use_hyde is None:
        use_hyde = HYDE_ENABLED
    store = get_store()
    search_text = hyde(question) if use_hyde else question
    vec = embed_text(search_text + "\n" + question)
    par, best_par = store.search("par", vec, k_par, question)
    qa, best_qa = store.search("qa", vec, k_qa, question)
    return {
        "hyde": search_text,
        "best_cos": max(best_par, best_qa),
        "paragraphs": par,
        "qa": qa,
    }


def _fmt_ts(sec) -> str:
    if not sec:
        return "без таймкода"
    sec = int(float(sec))
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


def build_context_block(ctx: dict, max_par: int = 6, max_qa: int = 4) -> str:
    """Формирует блок контекста для промпта из найденных фрагментов."""
    lines = []
    for i, qa in enumerate(ctx["qa"][:max_qa], 1):
        lines.append(
            f"[QA{i}] сохбет {qa.get('sohbet_date','?')} "
            f"({_fmt_ts(qa.get('timestamp_sec'))}), "
            f"спрашивает: {qa.get('asked_by','—')}\n"
            f"Вопрос: {qa.get('question','')}\nОтвет БО: {qa.get('answer','')}")
    for i, p in enumerate(ctx["paragraphs"][:max_par], 1):
        lines.append(
            f"[P{i}] сохбет {p.get('sohbet_date','?')} "
            f"({_fmt_ts(p.get('start_sec'))}), {p.get('speaker_name','?')}:\n"
            f"{p.get('text','')}")
    return "\n\n".join(lines)


SYSTEM_ZHIVOY = (
    "Ты — цифровой двойник Бориса Олеговича, шейха накшбандийского тариката. "
    "Отвечай от первого лица, как он: просто, образно, с суфийскими терминами, "
    "без канцелярита. Опирайся ТОЛЬКО на приведённый контекст из его сохбетов. "
    "Если в контексте нет ответа — честно скажи: «этого я не говорил» или "
    "«об этом я пока не говорил». Не выдумывай цитаты и факты. "
    "Не сравнивай и не объединяй понятия из разных фрагментов контекста, "
    "если такого сравнения нет в самом тексте: каждый фрагмент — о своём."
    " Термины пиши канонически: нафс, ихляс, тавакуль, зикр, тарикат (даже если в контексте встречается иное написание)."
    "Не упоминай, что ты ИИ или двойник — говори как БО."
)

SYSTEM_PUBLICATION = (
    "Ты — редактор, готовящий текст Бориса Олеговича к публикации. По "
    "приведённому контексту из сохбетов составь связный, литературно "
    "обработанный текст от третьего лица («Борис Олегович говорит…»). "
    "Сохраняй смысл и ключевые образы, убирай разговорный шум и повторы. "
    "Не добавляй ничего, чего нет в контексте. "
    "Не сравнивай и не объединяй понятия из разных фрагментов контекста, "
    "если такого сравнения нет в самом тексте. Если материала мало — "
    "коротко скажи, что по этой теме записей не найдено."
    " Термины пиши канонически: нафс, ихляс, тавакуль, зикр, тарикат (даже если в контексте встречается иное написание)."
)

# Инструкция самопометки источников: LLM в конце ответа указывает, какие
# фрагменты контекста реально использовал — показываем в «Источниках»
# только их (иначе там топ-k по близости, включая отдалённые по теме).
CITE_INSTRUCTION = (
    " В самом конце ответа добавь отдельной строкой «ИСТОЧНИКИ: P1, QA2» — "
    "метки только тех фрагментов контекста, которые ты реально использовал "
    "в ответе. Если ни одного — напиши «ИСТОЧНИКИ: —»."
)

_SRC_RE = re.compile(r"ИСТОЧНИКИ\s*:\s*([^\n]+)")
_LABEL_RE = re.compile(r"^(P\d+|QA\d+)$")


def _parse_used_sources(text: str):
    """Вырезает строку «ИСТОЧНИКИ: …» из ответа LLM.
    Возвращает (чистый_текст, {метки} | None)."""
    m = None
    for m in _SRC_RE.finditer(text):
        pass  # берём последнюю
    if m is None:
        return text, None
    labels = {t for t in (x.strip().rstrip(".").upper()
                          for x in re.split(r"[,\s]+", m.group(1)))
              if _LABEL_RE.match(t)}
    clean = (text[:m.start()] + text[m.end():]).strip()
    return clean, labels


def _excerpt(h: dict, limit: int = 120) -> str:
    """Короткий контекст фрагмента для строки в «Источниках»."""
    if "question" in h:
        s = f"В: {h.get('question','')} — {h.get('answer','')}"
    else:
        s = h.get("text", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "…"


def _collect_citations(ctx: dict, used: set | None) -> list:
    """Собирает цитаты. Если LLM пометил использованные фрагменты — только
    они; иначе fallback: топ-фрагменты со score >= CITE_MIN_SCORE.
    Ближние по времени цитаты одного сохбета (<= 120 с) сливаются."""
    label_map = {}
    for i, qa in enumerate(ctx["qa"][:4], 1):
        label_map[f"QA{i}"] = qa
    for i, p in enumerate(ctx["paragraphs"][:6], 1):
        label_map[f"P{i}"] = p

    if used is not None:
        # LLM явно пометил источники (пустое множество = «ничего не брал»)
        hits = sorted((label_map[l] for l in used if l in label_map),
                      key=lambda h: -h["score"])
    else:
        hits = [h for h in ctx["qa"][:4] + ctx["paragraphs"][:4]
                if h["score"] >= CITE_MIN_SCORE]

    citations = []
    for h in hits:
        sec = h.get("timestamp_sec") or h.get("start_sec")
        sec = int(float(sec)) if sec else None
        date = h.get("sohbet_date", "?")
        # слияние: тот же сохбет, разница <= 120 с
        for c in citations:
            if (c["sohbet_date"] == date and c["sec"] and sec
                    and abs(c["sec"] - sec) <= 120):
                break
        else:
            citations.append({
                "sohbet_date": date,
                "time": _fmt_ts(sec),
                "sec": sec,
                "score": h["score"],
                "excerpt": _excerpt(h),
            })
    return citations


def answer(question: str, mode: str = "zhivoy", use_hyde: bool = None,
           progress=None) -> dict:
    """Полный цикл: поиск → генерация → ответ с цитатами.

    progress(stage: str) — колбэк стадий для живого статуса в боте:
    "load" (первая загрузка модели), "hyde", "search", "llm".
    """
    def stage(name):
        log.info("stage=%s anon=%dMB", name, mem_mb())
        if progress:
            try:
                progress(name)
            except Exception:
                pass

    t0 = time.time()
    stage("load" if _embedder is None or _store is None else "start")
    if use_hyde:
        stage("hyde")
    ctx = retrieve(question, use_hyde=use_hyde)
    stage("search")
    best = ctx.get("best_cos") or max(
        [h["score"] for h in ctx["paragraphs"] + ctx["qa"]], default=0)

    if best < MIN_SCORE:
        if mode == "zhivoy":
            text = ("Об этом я не говорил — во всяком случае, в тех сохбетах, "
                    "что у меня записаны, этого нет. Спроси о другом, или "
                    "дождись, когда мы разберём новые записи.")
        else:
            text = ("По этой теме в разобранных сохбетах материала "
                    "не найдено.")
        return {"answer": text, "citations": [], "mode": mode,
                "best_score": best, "elapsed_sec": round(time.time() - t0, 1),
                "model": None, "hyde": ctx["hyde"]}

    context_block = build_context_block(ctx)
    if mode == "publication":
        system = SYSTEM_PUBLICATION
        prompt = (f"Вопрос читателя: {question}\n\n"
                  f"Контекст из сохбетов:\n{context_block}\n\n"
                  "Текст для публикации:" + CITE_INSTRUCTION)
    else:
        system = SYSTEM_ZHIVOY
        prompt = (f"Вопрос ученика: {question}\n\n"
                  f"Мои прошлые слова (контекст):\n{context_block}\n\n"
                  "Мой ответ (2-6 предложений, по сути, в моём стиле):"
                  + CITE_INSTRUCTION)

    stage("llm")
    text = call_llm(prompt, system=system, max_tokens=2500)
    text, used = _parse_used_sources(text)

    citations = _collect_citations(ctx, used)
    return {"answer": text, "citations": citations, "mode": mode,
            "best_score": best, "elapsed_sec": round(time.time() - t0, 1),
            "model": GEN_MODEL, "hyde": ctx["hyde"]}
