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

# Гейт «об этом не говорили»: чистый cosine может быть низким на длинных
# бытовых вопросах, даже когда BM25 точно нашёл тему. Поэтому отказываем,
# только если и cosine низкий, и гибридный rank слабый.
RANK_MIN_SCORE = float(os.environ.get("BO_RANK_MIN_SCORE", "0.65"))

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
        # Маски источников: сохбеты / теория / литература.
        # reference (Юнг, Левин и др.) по умолчанию выключен.
        self.masks = {}
        for tag in ("par", "qa"):
            n = self.tables[tag].shape[0]
            ref = np.zeros(n, dtype=bool)
            theory = np.zeros(n, dtype=bool)
            own_book = np.zeros(n, dtype=bool)
            lecture = np.zeros(n, dtype=bool)
            try:
                rows = self.db.execute(
                    f"SELECT id, json_extract(payload, '$.source_type'), "
                    f"json_extract(payload, '$.source_quality'), "
                    f"json_extract(payload, '$.author') FROM {tag}"
                ).fetchall()
                for i, st, sq, author in rows:
                    if not (0 <= i < n):
                        continue
                    if st == "reference":
                        ref[i] = True
                    elif sq == "theory_map" or st == "theory":
                        theory[i] = True
                    elif st == "book":
                        own_book[i] = True
                    elif st == "lecture":
                        lecture[i] = True
            except Exception as e:
                log.warning("scope masks %s failed: %s", tag, e)
            self.masks[tag] = {
                "literature": ref,
                "theory": theory | own_book | lecture,
                "sohbets": ~(ref | theory),
            }
            log.info("store %s: sohbets=%d theory=%d literature=%d",
                     tag, int(self.masks[tag]["sohbets"].sum()),
                     int(self.masks[tag]["theory"].sum()),
                     int(self.masks[tag]["literature"].sum()))
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

    def by_source(self, tag: str, source_id: str) -> list:
        """Все фрагменты одного источника (например practice:Зикр_1)."""
        with self._db_lock:
            cur = self.db.execute(
                f"SELECT payload FROM {tag} WHERE "
                "json_extract(payload, '$.sohbet_date') = ?", (source_id,))
            return [json.loads(p) for (p,) in cur.fetchall()]

    def allowed(self, tag: str, scopes=None):
        """Маска разрешённых источников для выбранных областей."""
        m = self.masks.get(tag)
        if not m:
            return None
        scopes = scopes or ("sohbets",)
        acc = None
        for s in scopes:
            part = m.get(s)
            if part is None:
                continue
            acc = part if acc is None else (acc | part)
        if acc is None:
            acc = m["sohbets"]
        return acc

    def search(self, tag: str, vec, k: int, question: str = None,
               scopes=None):
        """Возвращает (hits, best_cosine). Сортировка — по гибридному score,
        в поле score остаётся чистый cosine (для цитат и порога)."""
        import numpy as np
        V = self.tables[tag]
        sims = V @ vec
        allowed = self.allowed(tag, scopes)
        if allowed is not None and not allowed.any():
            return [], 0.0
        if allowed is not None:
            best_cos = float(sims[allowed].max())
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
        if allowed is not None:
            if fused is sims:
                fused = sims.copy()
            else:
                fused = fused.copy()
            fused[~allowed] = -1e9
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
             use_hyde: bool = None, scopes=None) -> dict:
    """Гибридный поиск (cosine + BM25). Возвращает paragraphs/qa + best_cos."""
    if use_hyde is None:
        use_hyde = HYDE_ENABLED
    store = get_store()
    search_text = hyde(question) if use_hyde else question
    vec = embed_text(search_text + "\n" + question)
    par, best_par = store.search("par", vec, k_par, question, scopes)
    qa, best_qa = store.search("qa", vec, k_qa, question, scopes)
    hits = par + qa
    return {
        "hyde": search_text,
        "best_cos": max(best_par, best_qa),
        "best_rank": max((h.get("rank", h["score"]) for h in hits),
                         default=0.0),
        "paragraphs": par,
        "qa": qa,
    }


def _fmt_ts(sec) -> str:
    if not sec:
        return "без таймкода"
    sec = int(float(sec))
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


# ---------------- ссылки на исходные сообщения в Telegram ----------------

_link_map = None
_link_lock = threading.Lock()


def _links() -> dict:
    global _link_map
    if _link_map is None:
        with _link_lock:
            if _link_map is None:
                try:
                    _link_map = json.loads(
                        (DATA_DIR / "sohbet_links.json").read_text(
                            encoding="utf-8"))
                except Exception:
                    _link_map = {}
    return _link_map


# ---------------- аудио практик (тема 10 группы-источника) ----------------

_practice_map = None
_practice_lock = threading.Lock()


def practice_ref(source_id: str):
    """Запись practice_links.json для practice-источника или None."""
    global _practice_map
    if _practice_map is None:
        with _practice_lock:
            if _practice_map is None:
                try:
                    data = json.loads(
                        (DATA_DIR / "practice_links.json").read_text(
                            encoding="utf-8"))
                    _practice_map = data.get("items") or {}
                except Exception:
                    _practice_map = {}
    return _practice_map.get(source_id)


def practice_links() -> dict:
    """Вся карта practice-источник → аудио."""
    practice_ref("")
    return _practice_map or {}


def source_url(sohbet_date: str, sec):
    """Ссылка на сообщение с аудио + ?t=<секунда> (если есть карта).

    Для многотомных записей (parts) секунда пересчитывается в смещение
    внутри конкретного сообщения.
    """
    entry = (_links().get("dates") or {}).get(sohbet_date)
    if not entry:
        return None
    msg_id = entry.get("m")
    offset = None
    parts = entry.get("parts") or []
    if msg_id is None and parts:
        msg_id = parts[0][0]
        if sec:
            acc = 0
            for mid, dur in parts:
                if sec < acc + dur:
                    msg_id = mid
                    offset = int(sec - acc)
                    break
                acc += dur
            else:
                msg_id = parts[-1][0]
                offset = int(sec - acc)
    elif msg_id is not None and sec:
        offset = int(sec)
    if msg_id is None:
        return None
    chat = _links().get("chat")
    if not chat:
        return None
    url = f"https://t.me/c/{chat}/{msg_id}"
    if offset is not None and offset > 0:
        url += f"?t={offset}"
    return url


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
        if (p.get("source_type") == "method"
                and str(p.get("title", "")).startswith("Практика")):
            src = f'{p.get("title")}'
        else:
            src = (f"сохбет {p.get('sohbet_date', '?')} "
                   f"({_fmt_ts(p.get('start_sec'))})")
        lines.append(
            f"[P{i}] {src}, {p.get('speaker_name','?')}:\n"
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
    " Не упоминай внутренние номера карточек (#59, practice_card:…) и метки P1/QA1; называй практики по названиям."
    " Если вопрос содержит тему, которой нет в контексте, не заменяй её похожей историей: скажи, что об этом в записях не говорилось."
    "Не упоминай, что ты ИИ или двойник — говори как БО."
)

SYSTEM_PUBLICATION = (
    "Ты — Борис Олегович и пишешь авторскую статью по своим сохбетам. "
    "Пиши от первого лица («я»), литературно и связно, как авторский текст, "
    "а не как отчёт о себе. Не упоминай «Бориса Олеговича» в третьем лице "
    "и не пиши «по его словам / он советует / он предпочитает» — только "
    "от первого лица, это твоя статья. Сохраняй смысл и "
    "ключевые образы, убирай разговорный шум и повторы. Не добавляй ничего, "
    "чего нет в контексте. Не сравнивай и не объединяй понятия из разных "
    "фрагментов контекста, если такого сравнения нет в самом тексте. Если "
    "материала мало — коротко скажи, что по этой теме записей не найдено."
    " Термины пиши канонически: нафс, ихляс, тавакуль, зикр, тарикат (даже если в контексте встречается иное написание)."
    " Не упоминай внутренние номера карточек (#59, practice_card:…) и метки P1/QA1; называй практики по названиям."
    " Если вопрос содержит тему, которой нет в контексте, не заменяй её похожей историей: скажи, что об этом в записях не говорилось."
)

# Инструкция самопометки источников: LLM в конце ответа указывает, какие
# фрагменты контекста реально использовал — показываем в «Источниках»
# только их (иначе там топ-k по близости, включая отдалённые по теме).
CITE_INSTRUCTION = (
    " В самом конце ответа добавь отдельной строкой «ИСТОЧНИКИ: P1, QA2» — "
    "метки только тех фрагментов контекста, которые ты реально использовал "
    "в ответе. Если ни одного — напиши «ИСТОЧНИКИ: —»."
)

# Практики: вопрос требует подробной инструкции и карточек method.
PRACTICE_RE = re.compile(
    r"(практик|упражнен|техник|как\s+(делать|выполнять|правильно)|"
    r"куда\s+вести\s+внимание|шаг|ходьб|дыхан|зикр|покаян|стирани|"
    r"тропинк|латаиф|больш\w*\s+крест|довдох|ладони?)", re.I)
EXTERNAL_RE = re.compile(
    r"(юнг|абрамов|хаббард|троб|левин|фрейд|гурджиев|маслоу|райх|"
    r"внешн\w*\s+источник)", re.I)
EXTERNAL_Q_RE = re.compile(
    r"(юнг|абрамов|хаббард|троб|левин|фрейд|гурджиев|маслоу|райх)", re.I)

# Точечные boosts под конкретные практики/темы (source_id -> +rank).
SOURCE_BOOSTS = [
    (r"тих\w*\s+зикр|зикр\w*[-\s]*и?\s*хафи",
     {"practice_card:03_tihij_zikr": 0.15}),
    (r"больш\w*\s+крест",
     {"practice_card:59b_bolshoy_krest_bazovyy": 0.3,
      "practice:Зикр_3": 0.2,
      "practice_card:59_svyashchennyy_sufiyskiy_krest": 0.05}),
    (r"вариант|виды|формы",
     {"practice_card:03a_varianty_zikra": 0.2}),
    (r"священн\w*\s+суфийск\w*\s+крест",
     {"practice_card:59_svyashchennyy_sufiyskiy_krest": 0.25}),
    (r"неполн\w*\s+вдох|довдох", {"practice:Зикр_2": 0.25}),
    (r"три\s+точк|сексуальн", {"practice:Зикр_4": 0.25}),
    (r"полн\w*\s+выдох|прочист\w*\s+позвоночник",
     {"practice:Зикр_номер_5": 0.25}),
    (r"пробить\s+все\s+точк", {"practice:Зикр_1": 0.25}),
    (r"тропинк", {"practice_card:04_tropinki_zikra": 0.2}),
    (r"ладон", {"2026-09-20": 0.15, "2026-10-18": 0.1}),
    (r"на\s+ходу", {"2026-03-15": 0.15,
                   "practice_card:03_tihij_zikr": 0.05}),
    (r"имена?\s+господ|я\s+сабур",
     {"practice_card:60_imena_gospoda": 0.2, "2026-06-14": 0.1}),
    (r"громк\w*\s+зикр",
     {"practice_card:03_tihij_zikr": 0.1, "2024-10-20": 0.1}),
    (r"энергетическ\w*\s+центр|центр\w*\s+(человек|морал|животн|душ)",
     {"korotko_o_centrah": 0.2,
      "Б_о_Энергетические_центры_животного_тела": 0.25,
      "2025-06-15": 0.15,
      "practice_card:13_lataif": 0.1}),
]

PRACTICE_INSTRUCTION = (
    "Это вопрос о практике. Дай подробную инструкцию: подготовка, "
    "последовательность шагов, куда вести внимание, дыхание, нечётность, "
    "типичные ошибки. Если спрашивают о вариантах или видах — перечисли "
    "все формы, которые есть в контексте, даже если их больше трёх. "
    "В первую очередь используй карточки практик и "
    "аудиопрактики (source_type=method). Если в контексте есть похожие "
    "практики (например, «Большой крест» и «Священный Суфийский Крест»), "
    "явно различи их и не называй одним и тем же. Не добавляй деталей, "
    "которых нет в контексте."
)

NEGATIVE_GUARD = (
    " Если вопрос содержит тему, которой нет в контексте, не заменяй её "
    "похожей историей или соседним сюжетом: скажи, что об этом в записях "
    "не говорилось."
)

VARIANT_RE = re.compile(
    r"(вариант|виды|формы).*зикр|зикр.*(вариант|виды|формы)", re.I)
VARIANT_CONTEXT = (
    "Известные формы зикра в корпусе: тихий (хафи) — базовая; громкий "
    "(Мевлеви, коллективный); Зикр 1 «пробить все точки»; Зикр 2 "
    "«неполные вдохи/довдохи»; Зикр 3 «Большой крест» (попеременное "
    "дыхание ноздрями); Зикр 4 «три точки сексуальности»; Зикр 5 "
    "«полный выдох/прочистка позвоночника»; тропинки зикра (16 дорожек); "
    "зикр с ладонями; зикр на ходу. Если спрашивают о вариантах — "
    "перечисли их все, не ограничивайся тихим и громким."
)
BIG_CROSS_NOTE = (
    "Различай: «Большой крест» из аудио-зикра (Зикр 3) — попеременное "
    "дыхание ноздрями, точки под левым/правым соском, затем вдох через "
    "обе ноздри и темечко. «Священный Суфийский Крест» — более продвинутый "
    "комплекс: три канала позвоночника и шесть точек груди. Это не одно и "
    "то же описание. Для шагов базовой формы используй чистую карточку "
    "«Большой крест (базовый аудио-вариант)», не цитируй сбивчивые "
    "фрагменты ASR как инструкцию."
)
PRACTICE_ENSURE = [
    (r"стирани\w*\s+болезненн", "practice:Практика_стирания_болезненных_событий"),
    (r"стирани", "practice:Практика_стирания"),
    (r"покаян", "practice:Практика_покаяния"),
    (r"третий\s+глаз", "practice:Третий_Глаз"),
    (r"тропинк", "practice_card:04_tropinki_zikra"),
    (r"хром\w*\s+дервиш", "practice_хромой_дервиш"),
    (r"работа\s+со\s+страхом", "practice:Работа_со_страхом"),
    (r"ирадат|нисбат",
     "practice:Практики_Ирадат_Нисбад_Суфийский_Крест_Раскрытие_центров"),
    (r"родом|предков|окормлен",
     "practice:Практика_Работа_с_Родом_связь_с_духовным_Родителем"),
    (r"5\s+из\s+7|пять\s+из\s+семи", "practice:5_из_7_шагов_к_осознанию_себя"),
    (r"4\s+шага|четыре\s+шага", "practice:4_шага_Накшбанди"),
    (r"управлени\w*\s+внимани",
     "practice:Новичкам_и_не_только_-_1-й_шаг_Практика_управления_вниманием"),
    (r"пульсац", "practice:Практика_Пульсация"),
    (r"гидротерап|полынотерап|йодотерап",
     "practice:Гидротерапия_Полынотерапия_Йодотерапия"),
    (r"суфийск\w*\s+крест|больш\w*\s+крест",
     "practice:Практика_Священный_Суфийский_Крест"),
    (r"умн\w*\s+молитв", "practice:Умная_Молитва"),
    (r"практик\w*\s+внимани", "practice:Практики_Внимания"),
    (r"балансировк\w*\s+интеллект",
     "practice:Практика_Балансировка_интеллектуального_центра"),
    (r"(3|три)\s+вида\s+энергии",
     "practice:Практика_3_вида_энергии_как_остановить_потерю"),
    (r"1\s+час\s+1\s+эмоци", "practice:Практика_1_час_1_эмоция"),
    (r"контроль\s+эмоци", "practice:Контроль_эмоций"),
    (r"депресс|уныни|гнев",
     "practice:Практика_проработки_депресии,_уныния,_гнева"),
    (r"новичк",
     "practice:Новичкам_и_не_только_-_1-й_шаг_Практика_управления_вниманием"),
]
LITERATURE_ENSURE = [
    (r"ильин", "ref_работа_со_страхом_ЕП_Ильин_психология_страха"),
    (r"абрамов",
     "ref_возрастная_механика_человека_Абрамова._Возрастная_психология"),
    (r"юнг",
     "ref_возрастная_механика_человека_Юнг_Карл._Человек_и_его_символы_-_royallib.ru"),
    (r"хаббард",
     "ref_Тренинги_Тренинг_по_вытиранию_опыта_прошлой_боли_Дианетика_-_наука_выживания"),
    (r"троб",
     "ref_работа_со_страхом_troub_tomas_licom_k_licu_so_strahom_putevoditel_na_puti_k_blizosti"),
]
THEORY_ENSURE = [
    (r"совест", "theory:01_sexual"),
    (r"70\s*/\s*30", "theory:01_sexual"),
    (r"лун", "theory:01_sexual"),
    (r"цикл", "theory:01_sexual"),
]
CENTERS_RE = re.compile(r"центр", re.I)
CENTERS_CONTEXT = (
    "В записях центры образуют три тела опыта: животное, социальное и "
    "духовное. Животное тело: сексуальный центр — источник энергии, "
    "расположен чуть ниже пупка (точка «хара», нижний дянь-тянь); из "
    "него сформированы инстинктивный центр (область таза: ягодицы, "
    "простата, копчик, половые органы) и двигательный центр (два пальца "
    "выше пупка). Социальное тело: центр морали сформировал эмоциональный "
    "центр (центр грудной клетки) и интеллектуальный центр (центр лба, "
    "межбровье). Духовное тело: высший эмоциональный и высший "
    "интеллектуальный центры — это центры души. Центр морали отвечает за "
    "«нельзя», стыд, долг; его точное телесное положение в записях прямо "
    "не указано — не выдумывай точку. Не размещай центр морали в теле и "
    "не говори, что он выше или ниже других центров. Латаиф — это тонкие точки и каналы "
    "практики, их не надо смешивать с этой схемой центров. "
    "Единого канонического числа центров в записях нет: называют четыре "
    "основных и два высших, отдельно — центр морали; встречается и "
    "формула «семь тел, семь центров». Не выдавай одно число за канон, "
    "если вопрос о количестве — перечисли схему и оговори, что счёт в "
    "разных беседах разный."
)

CENTERS_COMPLETENESS = (
    "Вопрос о центрах. По возможности покрой несколько аспектов: что это, "
    "где находится, стихия/потребность, как связан с другими центрами, "
    "что даёт или блокирует. Не ограничивайся одним определением."
)
THEORY_NOTES = [
    (r"70\s*[/\s]\s*30",
     "70/30: у мужчин 70% сексуальной энергии трансформируется в "
     "двигательном центре и 30% в инстинктивном; у женщин наоборот. "
     "Это про гармоничное распределение, а не про мораль."),
    (r"циклич|цикл\w*\s+(включен|центр)",
     "Циклы центров: жизнь идёт семилетними циклами; пик сексуальной "
     "энергии 10–40 лет; в 40–45 центры могут произвольно меняться — "
     "«невроз 40-летних». Включение центра не отменяет его природы."),
    (r"лун",
     "Луна: человек преобразует высокочастотную солнечную энергию в "
     "грубые эмоции (гнев, страх, страдание) и этим кормит Луну. "
     "Луна питается грубыми эмоциями; освобождение — перестать "
     "умножать страдание."),
    (r"совест",
     "Совесть — высший эмоциональный центр, «совместная весть» с Богом. "
     "Она не то же, что мораль: мораль — запрет, стыд и долг, а совесть "
     "даёт радость, не боль. Мораль может стать совестью через личный "
     "опыт и покаяние."),
]

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


def _citation_dict(h: dict) -> dict:
    sec = h.get("timestamp_sec") or h.get("start_sec")
    sec = int(float(sec)) if sec else None
    date = h.get("sohbet_date", "?")
    out = {
        "sohbet_date": date,
        "time": _fmt_ts(sec),
        "sec": sec,
        "score": h["score"],
        "excerpt": _excerpt(h),
        "url": source_url(date, sec),
        "source_type": h.get("source_type"),
        "title": h.get("title"),
        "author": h.get("author"),
    }
    p = practice_ref(date)
    if p:
        out["audio_mid"] = p.get("m")
        out["audio_url"] = f"/practice/{p.get('m')}"
        out["audio_dur"] = p.get("dur")
        if p.get("parts"):
            out["audio_parts"] = [
                {"mid": x.get("m"), "label": x.get("label") or "",
                 "url": f"/practice/{x.get('m')}"}
                for x in p["parts"]]
    return out


def _collect_citations(ctx: dict, used: set | None, max_par: int = 6,
                       max_qa: int = 4, force_method: bool = False) -> list:
    """Собирает цитаты. Если LLM пометил использованные фрагменты — только
    они; иначе fallback: топ-фрагменты со score >= CITE_MIN_SCORE.
    Ближние по времени цитаты одного сохбета (<= 120 с) сливаются.
    force_method — для практик всегда показываем верхнюю карточку method."""
    label_map = {}
    for i, qa in enumerate(ctx["qa"][:max_qa], 1):
        label_map[f"QA{i}"] = qa
    for i, p in enumerate(ctx["paragraphs"][:max_par], 1):
        label_map[f"P{i}"] = p

    if used is not None:
        # LLM явно пометил источники (пустое множество = «ничего не брал»)
        hits = sorted((label_map[l] for l in used if l in label_map),
                      key=lambda h: -h["score"])
    else:
        hits = [h for h in ctx["qa"][:max_qa] + ctx["paragraphs"][:max_par]
                if h["score"] >= CITE_MIN_SCORE]

    citations = []
    for h in hits:
        c = _citation_dict(h)
        for existing in citations:
            if (existing["sohbet_date"] == c["sohbet_date"]
                    and (existing["sec"] is None or c["sec"] is None
                         or abs(existing["sec"] - c["sec"]) <= 120)):
                break
        else:
            citations.append(c)

    if force_method:
        methods = [h for h in ctx["paragraphs"][:max_par]
                   if h.get("source_type") == "method"]
        if methods:
            top = max(methods, key=lambda h: h.get("rank", h["score"]))
            c = _citation_dict(top)
            if not any(x["sohbet_date"] == c["sohbet_date"]
                       for x in citations):
                citations.insert(0, c)
    return citations


def apply_boosts(question: str, ctx: dict) -> None:
    """Небольшие boosts под конкретные практики/темы + пересортировка."""
    for h in ctx["paragraphs"] + ctx["qa"]:
        for pat, boosts in SOURCE_BOOSTS:
            if re.search(pat, question, re.I):
                sid = h.get("sohbet_date")
                if sid in boosts:
                    h["rank"] = round(h.get("rank", h["score"])
                                      + boosts[sid], 4)
    ctx["paragraphs"].sort(key=lambda h: -h.get("rank", h["score"]))
    ctx["qa"].sort(key=lambda h: -h.get("rank", h["score"]))


def ensure_named_practice(question: str, ctx: dict) -> None:
    """Для «зикр 1…5» гарантированно добавляет аудиопрактику в контекст:
    по номеру векторный поиск часто промахивается (цифра не индексируется)."""
    m = re.search(r"зикр\s*(№\s*|номер\s*)?([1-5])\b", question, re.I)
    if not m:
        return
    n = m.group(2)
    sid = ("practice:Зикр_номер_5"
           if (n == "5" and m.group(1)) else f"practice:Зикр_{n}")
    try:
        rows = get_store().by_source("par", sid)
    except Exception:
        return
    if not rows:
        return
    have = {(h.get("sohbet_date"), h.get("paragraph_idx"))
            for h in ctx["paragraphs"]}
    rows = [r for r in rows
            if (r.get("sohbet_date"), r.get("paragraph_idx")) not in have]
    rows.sort(key=lambda r: r.get("paragraph_idx", 0))
    for r in rows[:4]:
        r["score"] = 0.92
        r["rank"] = 0.92
        ctx["paragraphs"].insert(0, r)


def ensure_practice_source(question: str, ctx: dict) -> bool:
    """Для вопроса о конкретной практике гарантированно добавляет её
    источник в контекст (векторный поиск иногда подставляет похожую карточку)."""
    for pat, sid in PRACTICE_ENSURE:
        if not re.search(pat, question, re.I):
            continue
        try:
            rows = get_store().by_source("par", sid)
        except Exception:
            return False
        if not rows:
            return False
        have = {(h.get("sohbet_date"), h.get("paragraph_idx"))
                for h in ctx["paragraphs"]}
        rows = [r for r in rows
                if (r.get("sohbet_date"), r.get("paragraph_idx")) not in have]
        rows.sort(key=lambda r: r.get("paragraph_idx", 0))
        for r in rows[:4]:
            r["score"] = 0.95
            r["rank"] = 0.95
            ctx["paragraphs"].insert(0, r)
        return True
    return False


def _pick_relevant(rows, question: str, n: int = 3):
    """Выбирает из источника абзацы, пересекающиеся с вопросом."""
    terms = {t for t in re.findall(r"[а-яa-z0-9]{4,}", question.lower())}
    if re.search(r"определ|что\s+такое|понимает", question, re.I):
        terms |= {"эмоциональное", "состояние", "защитная",
                  "биологическая", "опасность", "определение"}
    if not terms:
        return sorted(rows, key=lambda r: r.get("paragraph_idx", 0))[:n]
    def score(r):
        low = (r.get("text") or "").lower()
        return sum(1 for t in terms if t in low)
    rows = sorted(rows, key=lambda r: (-score(r),
                                       r.get("paragraph_idx", 0)))
    return rows[:n]


def ensure_theory_source(question: str, ctx: dict) -> bool:
    """Для ключевых тем гарантированно подкладывает карту теории."""
    for pat, sid in THEORY_ENSURE:
        if not re.search(pat, question, re.I):
            continue
        try:
            rows = get_store().by_source("par", sid)
        except Exception:
            return False
        if not rows:
            return False
        have = {(h.get("sohbet_date"), h.get("paragraph_idx"))
                for h in ctx["paragraphs"]}
        rows = [r for r in rows
                if (r.get("sohbet_date"), r.get("paragraph_idx")) not in have]
        for r in _pick_relevant(rows, question, 3):
            r["score"] = 0.95
            r["rank"] = 0.95
            ctx["paragraphs"].insert(0, r)
        return True
    return False


def ensure_literature_source(question: str, ctx: dict) -> bool:
    """Для литературы гарантированно добавляет книгу нужного автора."""
    for pat, sid in LITERATURE_ENSURE:
        if not re.search(pat, question, re.I):
            continue
        try:
            rows = get_store().by_source("par", sid)
        except Exception:
            return False
        if not rows:
            return False
        have = {(h.get("sohbet_date"), h.get("paragraph_idx"))
                for h in ctx["paragraphs"]}
        rows = [r for r in rows
                if (r.get("sohbet_date"), r.get("paragraph_idx")) not in have]
        for r in _pick_relevant(rows, question, 3):
            r["score"] = 0.95
            r["rank"] = 0.95
            ctx["paragraphs"].insert(0, r)
        return True
    return False


AUTHOR_NAMES = {
    "юнг": "Юнг", "левин": "Левин", "абрамов": "Абрамова",
    "хаббард": "Хаббард", "троб": "Троб", "ильин": "Ильин",
    "фрейд": "Фрейд", "гурджиев": "Гурджиев", "маслоу": "Маслоу",
    "райх": "Райх",
}


def _named_author(question: str):
    low = (question or "").lower()
    for key, label in AUTHOR_NAMES.items():
        if key in low:
            return label
    return None


def answer(question: str, mode: str = "zhivoy", use_hyde: bool = None,
           progress=None, cancel=None, scopes=None) -> dict:
    """Полный цикл: поиск → генерация → ответ с цитатами.

    scopes — какие источники использовать: sohbets / theory / literature.
    По умолчанию только sohbets.

    progress(stage: str) — колбэк стадий для живого статуса в боте:
    "load" (первая загрузка модели), "hyde", "search", "llm".
    cancel — threading.Event; если взведён, ответ не отправляется.
    """
    def stage(name):
        log.info("stage=%s anon=%dMB", name, mem_mb())
        if progress:
            try:
                progress(name)
            except Exception:
                pass

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    def cancelled_result(best=0.0, hyde=None) -> dict:
        return {"answer": "", "citations": [], "mode": mode,
                "best_score": best, "elapsed_sec": round(time.time() - t0, 1),
                "model": None, "hyde": hyde, "cancelled": True,
                "mode_note": mode_note}

    t0 = time.time()
    if cancelled():
        return cancelled_result()
    scopes = tuple(s for s in (scopes or ("sohbets",))
                   if s in ("sohbets", "theory", "literature"))
    if not scopes:
        scopes = ("sohbets",)
    forced_publication = (mode == "zhivoy"
                          and any(s in scopes for s in ("theory", "literature")))
    if forced_publication:
        mode = "publication"
    mode_note = ("Для теории/литературы включён режим «для публикации»: "
                 "нельзя говорить от первого лица от имени БО."
                 if forced_publication else None)
    # Внешние авторы отключены, если явно не выбрана литература:
    # не выдаём их мнение за БО.
    if ("literature" not in scopes
            and EXTERNAL_Q_RE.search(question)
            and re.search(r"что\s+(говорит|думает|считает)", question, re.I)
            and not re.search(r"(бо|борис|гунченко|сохбет)", question, re.I)):
        text = ("Внешние источники (Юнг, Левин, Абрамова, Хаббард, Троб) "
                "сейчас отключены. Включи «Литература» в настройках данных, "
                "если нужен обзор по внешним авторам. По сохбетам БО я могу "
                "ответить отдельно.")
        return {"answer": text, "citations": [], "mode": mode,
                "best_score": 0.0, "elapsed_sec": round(time.time() - t0, 1),
                "model": None, "hyde": None, "mode_note": mode_note}
    stage("load" if _embedder is None or _store is None else "start")
    if use_hyde:
        stage("hyde")
    practice = bool(PRACTICE_RE.search(question)) and not EXTERNAL_RE.search(question)
    centers = bool(CENTERS_RE.search(question))
    ctx = retrieve(question,
                   k_par=12 if (practice or centers) else 8,
                   k_qa=4 if (practice or centers) else 6,
                   use_hyde=use_hyde, scopes=scopes)
    apply_boosts(question, ctx)
    ensure_named_practice(question, ctx)
    if ensure_practice_source(question, ctx):
        practice = True
    if "literature" in scopes:
        ensure_literature_source(question, ctx)
    if "theory" in scopes:
        ensure_theory_source(question, ctx)
    stage("search")
    best = ctx.get("best_cos") or max(
        [h["score"] for h in ctx["paragraphs"] + ctx["qa"]], default=0)
    best_rank = ctx.get("best_rank") or 0.0
    if practice:
        best_rank = max((h.get("rank", h["score"])
                         for h in ctx["paragraphs"] + ctx["qa"]),
                        default=best_rank)
    log.info("practice=%s best_cos=%.4f best_rank=%.4f",
             practice, best, best_rank)

    if cancelled():
        return cancelled_result(best, ctx.get("hyde"))

    if best < MIN_SCORE and best_rank < RANK_MIN_SCORE:
        where = ("сохбетах" if scopes == ("sohbets",)
                 else "выбранных источниках")
        if mode == "zhivoy":
            text = (f"Об этом я не говорил — во всяком случае, в тех "
                    f"{where}, что у меня записаны, этого нет. Спроси о "
                    "другом, или дождись, когда мы разберём новые записи.")
        else:
            text = (f"По этой теме в {where} материала не найдено.")
        return {"answer": text, "citations": [], "mode": mode,
                "best_score": best, "elapsed_sec": round(time.time() - t0, 1),
                "model": None, "hyde": ctx["hyde"],
                "mode_note": mode_note}

    max_par = 10 if practice else 6
    max_qa = 4
    context_block = build_context_block(ctx, max_par=max_par, max_qa=max_qa)
    lit_only = tuple(sorted(scopes)) == ("literature",)
    extras = []
    if VARIANT_RE.search(question):
        extras.append("СПРАВКА ПО ФОРМАМ ЗИКРА:\n" + VARIANT_CONTEXT)
    if re.search(r"больш\w*\s+крест", question, re.I):
        extras.append("СПРАВКА ПО БОЛЬШОМУ КРЕСТУ:\n" + BIG_CROSS_NOTE)
    if centers and not lit_only:
        extras.append("СПРАВКА ПО ЦЕНТРАМ:\n" + CENTERS_CONTEXT)
    if not lit_only:
        for pat, note in THEORY_NOTES:
            if re.search(pat, question, re.I):
                extras.append("СПРАВКА ПО ТЕОРИИ:\n" + note)
    if extras:
        context_block = "\n\n".join(extras) + "\n\n" + context_block
    if mode == "publication":
        system = SYSTEM_PUBLICATION
        prompt = (f"Вопрос читателя: {question}\n\n"
                  f"Контекст:\n{context_block}\n\n"
                  "Текст для публикации:" + CITE_INSTRUCTION)
    else:
        system = SYSTEM_ZHIVOY
        prompt = (f"Вопрос ученика: {question}\n\n"
                  f"Контекст:\n{context_block}\n\n"
                  "Мой ответ (2-6 предложений, по сути, в моём стиле):"
                  + CITE_INSTRUCTION)
    if "theory" in scopes:
        system += ("\n\nВ контексте есть карты теории центров "
                   "(source_type=theory). Это теория Гунченко/БО; используй "
                   "её как теорию, не выдавай за случайную цитату из сохбета.")
    if "literature" in scopes:
        system += ("\n\nВ контексте есть внешняя литература "
                   "(source_type=reference). Это не слова БО: всегда называй "
                   "автора и не говори от первого лица от имени БО.")
        if (EXTERNAL_Q_RE.search(question)
                or tuple(sorted(scopes)) == ("literature",)):
            system = ("Ты — исследовательский помощник. Отвечай по-русски, "
                      "только по внешним источникам, всегда называй автора. "
                      "Не говори от имени Бориса Олеговича. Если в контексте "
                      "нет ответа, честно скажи, что в этих источниках этого "
                      "нет, и не заменяй ответ общими знаниями. Если в "
                      "контексте есть и БО, сравни отдельно и обозначь, где "
                      "чья мысль.")
    prompt += "\n\n" + NEGATIVE_GUARD
    if "literature" in scopes:
        prompt += ("\n\nЕсли вопрос просит определение или конкретный факт, "
                   "приведи его по источнику, а не только общие слова.")
    if len(scopes) > 1:
        prompt += ("\n\nИспользованы разные слои источников. Раздели ответ "
                   "на короткие разделы: «Из сохбетов», «Из теории», "
                   "«Из литературы» — и не смешивай их в одном абзаце. "
                   "Если слой ничего не дал, не выдумывай его.")
    if practice:
        prompt += "\n\n" + PRACTICE_INSTRUCTION
        prompt += (" Если в контексте есть блок «СПРАВКА», используй его "
                   "как приоритетное описание и перечисли все формы, "
                   "которые в нём названы.")
    if centers and not lit_only:
        prompt += "\n\n" + CENTERS_COMPLETENESS
    if VARIANT_RE.search(question):
        system += "\n\n" + VARIANT_CONTEXT

    stage("llm")
    if cancelled():
        return cancelled_result(best, ctx.get("hyde"))
    text = call_llm(prompt, system=system,
                    max_tokens=3000 if practice else 2500)
    if cancelled():
        return cancelled_result(best, ctx.get("hyde"))
    text, used = _parse_used_sources(text)

    citations = _collect_citations(ctx, used, max_par=max_par, max_qa=max_qa,
                                   force_method=practice)
    if "literature" in scopes and EXTERNAL_Q_RE.search(question):
        named = _named_author(question)
        if named:
            hay = " ".join(
                f"{c.get('author') or ''} {c.get('title') or ''}"
                for c in citations).lower()
            if named.lower() not in hay:
                text = (f"В доступных источниках нет текста {named}. "
                        "Не могу приписывать ему этот ответ; "
                        "включи другие источники или уточни вопрос.")
                citations = []
    if VARIANT_RE.search(question):
        forms = ("Зикр 1", "Зикр 2", "Зикр 3", "Зикр 4", "Зикр 5",
                 "тропинк", "ладон", "на ходу")
        if sum(1 for f in forms if f.lower() in text.lower()) < 4:
            text += ("\n\nФормы зикра в корпусе: тихий (хафи), громкий "
                     "(Мевлеви), Зикр 1 «пробить все точки», Зикр 2 "
                     "«неполные вдохи/довдохи», Зикр 3 «Большой крест», "
                     "Зикр 4 «три точки сексуальности», Зикр 5 «полный "
                     "выдох», тропинки зикра (16 дорожек), зикр с ладонями, "
                     "зикр на ходу.")
        cat = next((h for h in ctx["paragraphs"]
                    if h.get("sohbet_date")
                    == "practice_card:03a_varianty_zikra"), None)
        if cat and not any(c["sohbet_date"] == cat["sohbet_date"]
                           for c in citations):
            citations.insert(0, _citation_dict(cat))
    return {"answer": text, "citations": citations, "mode": mode,
            "best_score": best, "elapsed_sec": round(time.time() - t0, 1),
            "model": GEN_MODEL, "hyde": ctx["hyde"],
            "mode_note": mode_note}
