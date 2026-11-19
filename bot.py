#!/usr/bin/env python3
"""bot.py — Telegram-бот «цифровой двойник БО» для Render free tier (512 МБ).

Чистый stdlib (urllib + http.server), без aiogram/aiohttp:
импорт aiogram съедает ~200 МБ anon (тысячи pydantic-моделей Telegram API) —
при лимите 512 МБ это разница между «работает» и OOM-kill.

Особенности под Render:
  - web service обязан слушать $PORT → health-сервер на http.server
  - free tier засыпает через 15 мин без входящего трафика → self-ping
    каждые 10 минут на $RENDER_EXTERNAL_URL/health
  - секреты через env: BO_BOT_TOKEN, OPENCODE_GO_KEY
  - данные (плоский стор numpy + sqlite) и GGUF-модель скачиваются на build
    (fetch_assets.py из release-ассета), см. rag_core.py

Режимы: 🎙 Живой / 📝 Для публикации (инлайн-кнопки, per-user в памяти).
Debug: BO_DEBUG=1 — живое статус-сообщение по стадиям + anon-RSS в логах.
"""
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from html import escape as html_escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BOT_DIR))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bo_bot")

TOKEN = os.environ.get("BO_BOT_TOKEN", "").strip()
API = f"https://api.telegram.org/bot{TOKEN}"

MODE_NAMES = {"zhivoy": "🎙 Живой", "publication": "📝 Для публикации"}
user_mode: dict[int, str] = {}

SELF_PING_INTERVAL = 600  # 10 мин < 15 мин лимита Render
DEBUG = os.environ.get("BO_DEBUG") == "1"

# ---------------- доступ к Telegram-боту ----------------
# Пустой allowlist = доступ всем (локальная разработка).
# На проде: BO_ALLOWED_USERNAMES=alexenur,tgnamel
ALLOWED_USERNAMES = {
    u.strip().lstrip("@").lower()
    for u in os.environ.get("BO_ALLOWED_USERNAMES", "").split(",")
    if u.strip()
}
ALLOWED_IDS = {
    int(x) for x in os.environ.get("BO_ALLOWED_IDS", "").split(",")
    if x.strip().lstrip("-").isdigit()
}


def tg_allowed(user: dict) -> bool:
    """Разрешён ли пользователь. Пустые списки — доступ всем."""
    if not ALLOWED_USERNAMES and not ALLOWED_IDS:
        return True
    user = user or {}
    uname = (user.get("username") or "").lower()
    return uname in ALLOWED_USERNAMES or user.get("id") in ALLOWED_IDS


# ---------------- веб-чат ----------------
WEB_DIR = BOT_DIR / "web"
WEB_DAILY_LIMIT = int(os.environ.get("BO_WEB_DAILY_LIMIT", "40"))
WEB_PASSWORD = os.environ.get("BO_WEB_PASSWORD", "").strip()
# Голос: если заданы ключи Yandex SpeechKit — озвучиваем через него,
# иначе браузер делает это сам (Web Speech API).
YC_API_KEY = os.environ.get("YC_API_KEY", "").strip()
YC_FOLDER_ID = os.environ.get("YC_FOLDER_ID", "").strip()
YC_VOICE = os.environ.get("YC_VOICE", "filipp")
_web_hits: dict[str, list] = {}
_web_lock = threading.Lock()


def yandex_tts(text: str) -> bytes:
    data = urllib.parse.urlencode({
        "text": text[:1000],
        "lang": "ru-RU",
        "voice": YC_VOICE,
        "speed": "0.95",
        "format": "mp3",
        "folderId": YC_FOLDER_ID,
    }).encode()
    req = urllib.request.Request(
        "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
        data=data,
        headers={"Authorization": f"Api-Key {YC_API_KEY}",
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def web_rate_ok(ip: str) -> tuple[bool, int]:
    """(можно_ли, сколько_осталось). Счётчик — на календарный день."""
    today = time.strftime("%Y-%m-%d")
    with _web_lock:
        day, n = _web_hits.get(ip, [today, 0])
        if day != today:
            day, n = today, 0
        n += 1
        _web_hits[ip] = [day, n]
    return n <= WEB_DAILY_LIMIT, max(0, WEB_DAILY_LIMIT - n)

STAGE_TEXT = {
    "load": "⏳ Загружаю поисковую модель (первый раз — до минуты)…",
    "start": "⏳ Готовлю поиск…",
    "hyde": "🔍 Формулирую поисковый запрос…",
    "search": "📚 Ищу по записям сохбетов…",
    "llm": "🧠 Формулирую ответ…",
}


# ---------------- Telegram Bot API (urllib) ----------------

def tg(method: str, http_timeout: int = 40, **params) -> dict:
    """Вызов Bot API. Бросает RuntimeError при ошибке API."""
    body = json.dumps(params).encode()
    req = urllib.request.Request(
        f"{API}/{method}", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=http_timeout) as r:
        data = json.loads(r.read())
    if not data.get("ok"):
        raise RuntimeError(f"TG {method}: {str(data)[:200]}")
    return data["result"]


def send_message(chat_id: int, text: str, markup: dict = None) -> int:
    params = {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
              "disable_web_page_preview": True}
    if markup:
        params["reply_markup"] = markup
    return tg("sendMessage", **params)["message_id"]


def edit_message(chat_id: int, message_id: int, text: str,
                 markup: dict = None) -> None:
    params = {"chat_id": chat_id, "message_id": message_id, "text": text,
              "parse_mode": "HTML"}
    if markup:
        params["reply_markup"] = markup
    try:
        tg("editMessageText", **params)
    except Exception:
        pass  # «message is not modified» и прочее — не страшно


def delete_message(chat_id: int, message_id: int) -> None:
    try:
        tg("deleteMessage", chat_id=chat_id, message_id=message_id)
    except Exception:
        pass


def send_long(chat_id: int, text: str) -> None:
    chunk = ""
    for para in text.split("\n"):
        if len(chunk) + len(para) + 1 > 4000:
            send_message(chat_id, chunk)
            chunk = ""
        chunk += para + "\n"
    if chunk.strip():
        send_message(chat_id, chunk)


# ---------------- группа-источник (карта сохбетов) ----------------

# id группы «my | Сохбеты» (env BO_SOURCE_CHAT_ID). Нужен для copyMessage
# при тесте кликабельных таймкодов и для будущих ссылок t.me/c/<id>/<msg>.
SOURCE_CHAT_ID = os.environ.get("BO_SOURCE_CHAT_ID", "").strip()

MEDIA_KEYS = ("audio", "voice", "document", "video", "video_note")


def log_group_message(msg: dict) -> bool:
    """Логируем сообщения группы-источника: Bot API не умеет читать историю
    группы, поэтому новые МП3 регистрируем по мере появления — по этим
    строкам лога потом собирается карта сохбет → (chat_id, message_id).
    Возвращает True, если сообщение из группы (его не обрабатываем как вопрос).
    """
    chat = msg.get("chat") or {}
    if chat.get("type") not in ("group", "supergroup"):
        return False
    kind, fname = "text", ""
    for k in MEDIA_KEYS:
        media = msg.get(k)
        if media:
            kind = k
            fname = media.get("file_name") or ""
            break
    log.info("GROUP_MSG chat_id=%s msg_id=%s kind=%s file=%r title=%r",
             chat.get("id"), msg.get("message_id"), kind, fname,
             chat.get("title"))
    return True


def fmt_ts(seconds: int) -> str:
    h, rem = divmod(max(0, int(seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# Карта «дата сохбета → message_id в группе-источнике» (data/sohbet_links.json,
# скачивается fetch_assets.py). Ссылка t.me/c/… открывается только
# у участников группы — это и нужно для проверки ответов.
LINKS_PATH = BOT_DIR / "data" / "sohbet_links.json"
_link_map: dict | None = None


def source_link(sohbet_date: str, sec) -> str | None:
    """URL на аудио сохбета в группе. None, если даты нет в карте.
    Многочастевые записи: по офсету выбираем нужную часть (кумулятивно)."""
    global _link_map
    if _link_map is None:
        try:
            _link_map = json.loads(
                LINKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _link_map = {}
    entry = (_link_map.get("dates") or {}).get(sohbet_date)
    if not entry:
        return None
    msg_id = entry.get("m")
    parts = entry.get("parts") or []
    if msg_id is None and parts:
        msg_id = parts[0][0]
        if sec:
            acc = 0
            for mid, dur in parts:
                if sec < acc + dur:
                    msg_id = mid
                    break
                acc += dur
            else:
                msg_id = parts[-1][0]
    if msg_id is None:
        return None
    return f"https://t.me/c/{_link_map.get('chat')}/{msg_id}"


def source_audio_ref(sohbet_date: str, sec):
    """(msg_id, секунды_в_файле) для аудио источника или None.
    Для многочастевых записей вычитает длительность предыдущих частей —
    таймкод в подписи ведёт на позицию внутри конкретного файла."""
    global _link_map
    if _link_map is None:
        try:
            _link_map = json.loads(
                LINKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _link_map = {}
    entry = (_link_map.get("dates") or {}).get(sohbet_date)
    if not entry or sec is None:
        return None
    if entry.get("m"):
        return entry["m"], int(sec)
    parts = entry.get("parts") or []
    acc = 0
    for mid, dur in parts:
        if sec < acc + dur:
            return mid, int(sec - acc)
        acc += dur
    if parts:
        return parts[-1][0], int(sec - acc + parts[-1][1])
    return None


def send_top_source_audio(chat_id: int, res: dict) -> None:
    """Одно сообщение с аудио основного источника: таймкод в подписи
    кликабелен и запускает плеер с нужной секунды (проверено /srctest)."""
    if not SOURCE_CHAT_ID:
        return
    for c in res.get("citations", []):
        ref = source_audio_ref(c["sohbet_date"], c.get("sec"))
        if not ref:
            continue
        msg_id, sec = ref
        tg("copyMessage", chat_id=chat_id,
           from_chat_id=int(SOURCE_CHAT_ID), message_id=msg_id,
           caption=(f"▶️ Основной источник: сохбет {c['sohbet_date']}, "
                    f"слушайте с {fmt_ts(sec)}"))
        return


def handle_srctest(chat_id: int, text: str) -> None:
    """Живой тест кликабельных таймкодов: /srctest <msg_id> [секунды] [режим]

    Режимы:
      reply (по умолчанию) — copyMessage аудио + отдельный реплай с таймкодом
                             (подтверждён работающим: таймкод кликабелен и
                             запускает плеер с нужной секунды);
      caption — copyMessage, но таймкод в ПОДПИСИ к аудио (одно сообщение);
      caption_multi — в подписи несколько таймкодов (проверка списка);
      text — контроль: просто текст с таймкодом без привязки к аудио.
    """
    if not SOURCE_CHAT_ID:
        send_message(chat_id, "BO_SOURCE_CHAT_ID не задан — сначала нужен "
                              "chat_id группы (виден в логах Render после "
                              "любого сообщения в группе).")
        return
    parts = text.split()
    if len(parts) < 2 or not parts[1].isdigit():
        send_message(chat_id, "Формат: /srctest <msg_id> [секунды] "
                              "[reply|caption|caption_multi|text]")
        return
    msg_id = int(parts[1])
    sec = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    mode = parts[3] if len(parts) > 3 else "reply"
    ts = fmt_ts(sec)

    if mode == "text":
        send_message(chat_id, f"Контроль: таймкод {ts} просто текстом, "
                              "без привязки к аудио.")
        return

    if mode in ("caption", "caption_multi"):
        if mode == "caption":
            caption = f"▶️ Слушайте с {ts}"
        else:
            sec2 = sec + 600
            caption = (f"Проверка списка таймкодов в подписи:\n"
                       f"▶️ Фрагмент 1 — {ts}\n"
                       f"▶️ Фрагмент 2 — {fmt_ts(sec2)}")
        tg("copyMessage", chat_id=chat_id,
           from_chat_id=int(SOURCE_CHAT_ID), message_id=msg_id,
           caption=caption)
        return

    copied = tg("copyMessage", chat_id=chat_id,
                from_chat_id=int(SOURCE_CHAT_ID), message_id=msg_id)
    tg("sendMessage", chat_id=chat_id, text=f"▶️ Слушайте с {ts}",
       reply_to_message_id=copied["message_id"],
       allow_sending_without_reply=True)


# ---------------- логика бота ----------------

def mode_keyboard(current: str) -> dict:
    def btn(mode: str) -> dict:
        mark = "✅ " if mode == current else ""
        return {"text": mark + MODE_NAMES[mode],
                "callback_data": f"mode:{mode}"}
    return {"inline_keyboard": [[btn("zhivoy")], [btn("publication")]]}


def get_mode(user_id: int) -> str:
    return user_mode.get(user_id, "zhivoy")


def format_answer(res: dict) -> str:
    text = res["answer"]
    if res["citations"]:
        text += "\n\n— Источники:"
        for c in res["citations"]:
            label = f"сохбет {c['sohbet_date']}"
            url = source_link(c["sohbet_date"], c.get("sec"))
            if url:
                label = f'<a href="{url}">{label}</a>'
            line = f"\n• {label}, {c['time']}"
            if c.get("excerpt"):
                line += f" — <i>{html_escape(c['excerpt'])}</i>"
            text += line
    if DEBUG:
        text += (f"\n\n<i>debug: {res['elapsed_sec']}s, "
                 f"best={res['best_score']}, model={res['model']}</i>")
    return text


def handle_question(msg: dict) -> None:
    chat_id = msg["chat"]["id"]
    user_id = msg["from"]["id"]
    mode = get_mode(user_id)
    question = (msg.get("text") or "").strip()
    if len(question) < 3:
        return
    try:
        tg("sendChatAction", chat_id=chat_id, action="typing")
    except Exception:
        pass
    t0 = time.time()
    log.info("Q from %s mode=%s: %s", user_id, mode, question[:80])

    status_id = send_message(chat_id, "⏳ …")
    last_edit = [0.0]

    def progress(stage: str) -> None:
        if time.time() - last_edit[0] < 2:  # не чаще раза в 2 сек
            return
        last_edit[0] = time.time()
        txt = STAGE_TEXT.get(stage, stage)
        if DEBUG:
            from rag_core import mem_mb
            txt += (f"\n<i>debug: stage={stage}, {time.time()-t0:.0f}s, "
                    f"anon={mem_mb()}MB</i>")
        edit_message(chat_id, status_id, txt)

    try:
        from rag_core import answer
        res = answer(question, mode, progress=progress)
    except Exception as e:
        log.exception("rag error")
        delete_message(chat_id, status_id)
        send_message(chat_id, "Произошла ошибка при поиске ответа. "
                              f"Попробуй ещё раз чуть позже.\n({e})")
        return
    delete_message(chat_id, status_id)
    send_long(chat_id, format_answer(res))
    try:
        send_top_source_audio(chat_id, res)
    except Exception as e:
        log.warning("top source audio failed: %s", e)
    log.info("A in %ss best=%s", res["elapsed_sec"], res["best_score"])


def handle_update(upd: dict) -> None:
    if "callback_query" in upd:
        cq = upd["callback_query"]
        if not tg_allowed(cq.get("from")):
            try:
                tg("answerCallbackQuery", callback_query_id=cq["id"],
                   text="Доступ ограничен.")
            except Exception:
                pass
            return
        data = cq.get("data", "")
        if data.startswith("mode:"):
            mode = data.split(":", 1)[1]
            user_mode[cq["from"]["id"]] = mode
            msg = cq["message"]
            edit_message(msg["chat"]["id"], msg["message_id"],
                         f"Режим: {MODE_NAMES[mode]}", mode_keyboard(mode))
        try:
            tg("answerCallbackQuery", callback_query_id=cq["id"])
        except Exception:
            pass
        return

    msg = upd.get("message") or {}
    if log_group_message(msg):
        return  # в группе-источнике только регистрируем, не отвечаем
    text = (msg.get("text") or "").strip()
    if not text:
        return
    chat_id = msg["chat"]["id"]
    user_id = msg["from"]["id"]

    if not tg_allowed(msg.get("from")):
        log.info("ACCESS DENIED user_id=%s username=%s",
                 user_id, (msg.get("from") or {}).get("username"))
        send_message(chat_id, "Доступ к этому боту ограничен.")
        return

    if text.startswith("/start"):
        send_message(chat_id,
                     "Ассаляму алейкум. Я — цифровой двойник Бориса Олеговича, "
                     "отвечаю по записям его сохбетов.\n\n"
                     "Просто напиши вопрос. Режим ответа можно переключить:",
                     mode_keyboard(get_mode(user_id)))
    elif text.startswith("/mode"):
        send_message(chat_id,
                     f"Текущий режим: {MODE_NAMES[get_mode(user_id)]}",
                     mode_keyboard(get_mode(user_id)))
    elif text.startswith("/stats"):
        from rag_core import get_store
        c = get_store().counts()
        send_message(chat_id, f"База: {c['par']} фрагментов сохбетов, "
                              f"{c['qa']} пар вопрос-ответ.")
    elif text.startswith("/srctest"):
        try:
            handle_srctest(chat_id, text)
        except Exception as e:
            log.exception("srctest error")
            send_message(chat_id, f"srctest не сработал: {e}")
    elif not text.startswith("/"):
        handle_question(msg)


def polling_loop() -> None:
    offset = 0
    log.info("bot starting (long polling), debug=%s", DEBUG)
    while True:
        try:
            updates = tg("getUpdates", http_timeout=60, offset=offset,
                         timeout=45,
                         allowed_updates=["message", "callback_query"])
        except urllib.error.HTTPError as e:
            log.error("getUpdates HTTP %s — сплю 5с", e.code)
            time.sleep(5)
            continue
        except Exception as e:
            log.error("getUpdates %s — сплю 3с", e)
            time.sleep(3)
            continue
        for upd in updates:
            offset = upd["update_id"] + 1
            try:
                handle_update(upd)
            except Exception:
                log.exception("update handler error")


# ---------------- Render: HTTP (веб-чат + health) + анти-сон ----------------

class WebHandler(BaseHTTPRequestHandler):
    """GET / — веб-чат, GET /health, GET /stats, POST /ask."""

    server_version = "BO-RAG/1.0"

    def _send(self, code: int, body, ctype: str = "application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
            except Exception:
                page = "<h1>BO RAG</h1><p>web/index.html не найден</p>"
            self._send(200, page, "text/html; charset=utf-8")
        elif path == "/health":
            self._send(200, json.dumps(
                {"status": "ok", "bot": "bo_rag_bot"}))
        elif path == "/stats":
            try:
                from rag_core import get_store
                c = get_store().counts()
                self._send(200, json.dumps({"status": "ok", **c}))
            except Exception as e:
                self._send(500, json.dumps(
                    {"status": "error", "message": str(e)[:200]}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    do_HEAD = do_GET

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/tts":
            if not (YC_API_KEY and YC_FOLDER_ID):
                self._send(501, json.dumps({
                    "error": "tts",
                    "message": "Yandex SpeechKit не настроен — используйте голос браузера.",
                }, ensure_ascii=False))
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
                text = (payload.get("text") or "").strip()[:1000]
                if not text:
                    raise ValueError("empty")
                audio = yandex_tts(text)
                self._send(200, audio, "audio/mpeg")
            except Exception as e:
                log.warning("tts failed: %s", e)
                self._send(500, json.dumps(
                    {"error": "tts", "message": str(e)[:200]},
                    ensure_ascii=False))
            return

        if path != "/ask":
            self._send(404, json.dumps({"error": "not found"}))
            return

        ip = (self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
              or self.client_address[0])
        ok, left = web_rate_ok(ip)
        if not ok:
            self._send(429, json.dumps({
                "error": "limit",
                "message": "Дневной лимит вопросов исчерпан. Попробуйте завтра.",
            }, ensure_ascii=False))
            return

        if WEB_PASSWORD:
            if self.headers.get("X-Access-Code", "").strip() != WEB_PASSWORD:
                self._send(401, json.dumps(
                    {"error": "auth", "message": "Неверный код доступа."},
                    ensure_ascii=False))
                return

        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            question = (payload.get("question") or "").strip()[:1000]
            mode = payload.get("mode") or "publication"
            if mode not in ("zhivoy", "publication"):
                mode = "publication"
        except Exception:
            self._send(400, json.dumps(
                {"error": "bad request"}, ensure_ascii=False))
            return

        if len(question) < 3:
            self._send(400, json.dumps(
                {"error": "empty", "message": "Вопрос слишком короткий."},
                ensure_ascii=False))
            return

        t0 = time.time()
        log.info("WEB Q ip=%s mode=%s: %s", ip, mode, question[:90])
        try:
            from rag_core import answer
            res = answer(question, mode)
            body = {
                "answer": res["answer"],
                "citations": res["citations"],
                "mode": res["mode"],
                "best_score": res["best_score"],
                "elapsed_sec": res["elapsed_sec"],
                "left": left,
            }
            self._send(200, json.dumps(body, ensure_ascii=False))
            log.info("WEB A ip=%s %ss best=%s", ip,
                     round(time.time() - t0, 1), res["best_score"])
        except Exception as e:
            log.exception("web /ask error")
            self._send(500, json.dumps({
                "error": "internal",
                "message": f"Ошибка поиска: {str(e)[:200]}",
            }, ensure_ascii=False))

    def log_message(self, fmt, *args):
        # health/статику не логируем, чтобы не засорять лог
        if self.path and not self.path.startswith(("/health", "/", "/index")):
            log.info("HTTP %s", fmt % args)


def start_http_server() -> None:
    port = int(os.environ.get("PORT", "10000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), WebHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("HTTP server on :%s", port)


def self_ping_loop() -> None:
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        log.info("RENDER_EXTERNAL_URL не задан — self-ping выключен")
        return
    target = url.rstrip("/") + "/health"
    while True:
        time.sleep(SELF_PING_INTERVAL)
        try:
            with urllib.request.urlopen(target, timeout=30) as r:
                log.info("self-ping %s -> %s", target, r.status)
        except Exception as e:
            log.warning("self-ping failed: %s", e)


def main() -> None:
    start_http_server()  # порт сразу — иначе Render убьёт по таймауту
    threading.Thread(target=self_ping_loop, daemon=True).start()
    if os.environ.get("BO_WEB_ONLY") == "1":
        log.info("BO_WEB_ONLY=1 — Telegram polling выключен")
        while True:
            time.sleep(3600)
    if not TOKEN:
        sys.exit("Нет токена: env BO_BOT_TOKEN")
    polling_loop()


if __name__ == "__main__":
    main()
