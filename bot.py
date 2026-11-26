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
SCOPE_NAMES = {"sohbets": "📚 Сохбеты", "theory": "🧠 Теория",
               "literature": "📖 Литература"}
user_mode: dict[int, str] = {}
user_voice: dict[int, str] = {}
user_speed: dict[int, float] = {}
user_tts: dict[int, bool] = {}
user_scopes: dict[int, set] = {}
cancel_events: dict[int, threading.Event] = {}
SETTINGS_PATH = BOT_DIR / "data" / "user_settings.json"


def load_settings() -> None:
    try:
        d = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    for k, v in (d.get("mode") or {}).items():
        user_mode[int(k)] = v
    for k, v in (d.get("voice") or {}).items():
        user_voice[int(k)] = v
    for k, v in (d.get("speed") or {}).items():
        user_speed[int(k)] = float(v)
    for k, v in (d.get("tts") or {}).items():
        user_tts[int(k)] = bool(v)
    for k, v in (d.get("scopes") or {}).items():
        user_scopes[int(k)] = {s for s in v if s in SCOPE_NAMES}
    log.info("settings loaded: voice=%d speed=%d tts=%d scopes=%d",
             len(user_voice), len(user_speed), len(user_tts),
             len(user_scopes))


def save_settings() -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps({
            "mode": {str(k): v for k, v in user_mode.items()},
            "voice": {str(k): v for k, v in user_voice.items()},
            "speed": {str(k): v for k, v in user_speed.items()},
            "tts": {str(k): v for k, v in user_tts.items()},
            "scopes": {str(k): sorted(v) for k, v in user_scopes.items()},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        log.warning("save settings failed: %s", e)

SELF_PING_INTERVAL = 600  # 10 мин < 15 мин лимита Render
DEBUG = os.environ.get("BO_DEBUG") == "1"

# Webhook-режим для Telegram (надёжнее polling на free tier: не конфликтует
# с другими getUpdates и не требует постоянно живого процесса).
WEBHOOK_SECRET = os.environ.get("BO_WEBHOOK_SECRET", "").strip()
USE_WEBHOOK = (os.environ.get("BO_USE_WEBHOOK") == "1"
               and bool(WEBHOOK_SECRET))
_tg_lock = threading.Lock()

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
# Голос: Yandex SpeechKit (официальный, «как Алиса») при наличии ключей,
# иначе Edge TTS (бесплатные русские нейроголоса Microsoft), иначе браузер.
YC_API_KEY = (os.environ.get("YC_API_KEY")
              or os.environ.get("YANDEX_CLOUD_API_KEY", "")).strip()
YC_FOLDER_ID = (os.environ.get("YC_FOLDER_ID")
                or os.environ.get("YANDEX_CLOUD_FOLDER_ID", "")).strip()
YC_SA_ID = os.environ.get("YC_SA_ID", "").strip()
YC_KEY_ID = os.environ.get("YC_KEY_ID", "").strip()
YC_PRIVATE_KEY = os.environ.get("YC_PRIVATE_KEY", "").replace("\\n", "\n").strip()
YC_VOICE = os.environ.get("YC_VOICE", "filipp")
EDGE_VOICE = os.environ.get("BO_EDGE_VOICE", "ru-RU-DmitryNeural")
TTS_PROVIDER = os.environ.get("BO_TTS_PROVIDER", "auto").strip().lower()
TTS_CACHE = BOT_DIR / "data" / "tts_cache"
_web_hits: dict[str, list] = {}
_web_lock = threading.Lock()
_yc_token = None
_yc_token_exp = 0.0
_yc_lock = threading.Lock()
_yc_fail_until = 0.0

YANDEX_VOICES = [
    {"id": "filipp", "label": "Филипп — спокойный мужской"},
    {"id": "ermil", "label": "Ермил — декламационный мужской"},
    {"id": "zahar", "label": "Захар — универсальный мужской"},
    {"id": "alena", "label": "Алена — как Алиса"},
    {"id": "jane", "label": "Джейн — спокойный женский"},
    {"id": "omazh", "label": "Омаж — женский"},
    {"id": "dasha", "label": "Даша — женский"},
]
EDGE_VOICES = [
    {"id": "ru-RU-DmitryNeural", "label": "Дмитрий — мужской"},
    {"id": "ru-RU-SvetlanaNeural", "label": "Светлана — женский"},
    {"id": "ru-RU-DariyaNeural", "label": "Дария — женский"},
]
_YANDEX_IDS = {v["id"] for v in YANDEX_VOICES}
_EDGE_IDS = {v["id"] for v in EDGE_VOICES}


def yandex_ready() -> bool:
    if TTS_PROVIDER == "edge" or time.time() < _yc_fail_until:
        return False
    return bool(YC_FOLDER_ID and (
        YC_API_KEY or (YC_SA_ID and YC_KEY_ID and YC_PRIVATE_KEY)))


def yandex_iam_token() -> str:
    """IAM-токен по JWT сервисного аккаунта (живёт ~12 ч, кэшируем 50 мин)."""
    global _yc_token, _yc_token_exp
    with _yc_lock:
        if _yc_token and time.time() < _yc_token_exp:
            return _yc_token
        import jwt  # PyJWT[crypto]
        now = int(time.time())
        payload = {
            "iss": YC_SA_ID,
            "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            "iat": now,
            "exp": now + 3600,
        }
        signed = jwt.encode(payload, YC_PRIVATE_KEY, algorithm="PS256",
                            headers={"kid": YC_KEY_ID})
        req = urllib.request.Request(
            "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            data=json.dumps({"jwt": signed}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        _yc_token = data["iamToken"]
        _yc_token_exp = time.time() + 50 * 60
        return _yc_token


def resolve_voice(v: str = "") -> str:
    """ID голоса или male/female → конкретный голос провайдера.

    По умолчанию — мужской (БО мужчина): Yandex filipp / Edge Dmitry.
    """
    v = (v or "").strip()
    low = v.lower()
    male_hint = low in ("male", "m", "мужской") or any(
        x in low for x in ("dmitry", "filipp", "philipp", "ermil", "zahar",
                           "max", "nicolay", "kostya", "pavel"))
    female_hint = low in ("female", "f", "женский") or any(
        x in low for x in ("alena", "svetlana", "dariya", "jane", "omazh",
                           "dasha", "masha", "yulya", "julia"))
    male = male_hint or not female_hint
    if yandex_ready():
        if v in _YANDEX_IDS:
            return v
        return "filipp" if male else "alena"
    if v in _EDGE_IDS:
        return v
    return "ru-RU-DmitryNeural" if male else "ru-RU-SvetlanaNeural"


def yandex_tts(text: str, voice: str = None, speed: float = 0.95,
               fmt: str = "mp3") -> bytes:
    global _yc_fail_until
    if YC_API_KEY:
        auth = f"Api-Key {YC_API_KEY}"
    elif YC_SA_ID and YC_KEY_ID and YC_PRIVATE_KEY:
        auth = f"Bearer {yandex_iam_token()}"
    else:
        raise RuntimeError("Yandex SpeechKit не настроен")
    speed = max(0.5, min(1.5, float(speed or 0.95)))
    params = {
        "text": text[:1000],
        "lang": "ru-RU",
        "voice": voice or YC_VOICE,
        "speed": f"{speed:.2f}",
        "format": fmt,
        "folderId": YC_FOLDER_ID,
    }
    if fmt == "oggopus":
        params["sampleRateHertz"] = "48000"
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(
        "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
        data=data,
        headers={"Authorization": auth,
                 "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            _yc_fail_until = time.time() + 3600
            log.warning("Yandex TTS auth failed (%s) — Edge fallback 1h",
                        e.code)
        raise


def edge_tts_mp3(text: str, voice: str = None, speed: float = 0.95) -> bytes:
    """Бесплатный русский нейроголос Microsoft Edge (без ключа)."""
    import asyncio
    import hashlib

    import edge_tts

    voice = voice or EDGE_VOICE
    speed = max(0.5, min(1.5, float(speed or 0.95)))
    rate = f"{(speed - 1) * 100:+.0f}%"
    key = hashlib.sha1(f"{voice}\n{rate}\n{text}".encode("utf-8")).hexdigest()
    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    path = TTS_CACHE / f"{key}.mp3"
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()

    async def run():
        comm = edge_tts.Communicate(text[:1000], voice, rate=rate)
        await comm.save(str(path))

    asyncio.run(run())
    return path.read_bytes()


def get_voice(uid: int) -> str:
    return user_voice.get(uid) or resolve_voice("")


def get_speed(uid: int) -> float:
    return float(user_speed.get(uid, 0.95))


def get_tts(uid: int) -> bool:
    return bool(user_tts.get(uid, False))


def get_scopes(uid: int) -> tuple:
    s = user_scopes.get(uid) or {"sohbets"}
    return tuple(sorted(s))


def scopes_label(uid: int) -> str:
    names = [SCOPE_NAMES[s].split(" ", 1)[1]
             for s in get_scopes(uid) if s in SCOPE_NAMES]
    return " + ".join(names) if names else "Сохбеты"


def voice_label(vid: str) -> str:
    for v in YANDEX_VOICES + EDGE_VOICES:
        if v["id"] == vid:
            return v["label"].split(" — ")[0]
    return vid


def settings_keyboard(uid: int) -> dict:
    mode = get_mode(uid)
    voice = get_voice(uid)
    speed = get_speed(uid)
    tts = get_tts(uid)

    def mark(ok):
        return "✅ " if ok else ""

    return {"inline_keyboard": [
        [{"text": mark(mode == "zhivoy") + "🎙 Живой",
          "callback_data": "set:mode:zhivoy"},
         {"text": mark(mode == "publication") + "📝 Публикация",
          "callback_data": "set:mode:publication"}],
        [{"text": f"🎤 Голос: {voice_label(voice)}",
          "callback_data": "menu:voice"}],
        [{"text": f"⚡ Скорость: {speed:.2f}×",
          "callback_data": "menu:speed"}],
        [{"text": ("🔊 Озвучка ответов: вкл" if tts
                   else "🔇 Озвучка ответов: выкл"),
          "callback_data": "set:tts:" + ("off" if tts else "on")}],
        [{"text": f"📚 Данные: {scopes_label(uid)}",
          "callback_data": "menu:data"}],
        [{"text": "▶️ Проверить голос", "callback_data": "test:voice"}],
    ]}


def voice_menu_keyboard(uid: int) -> dict:
    voices = YANDEX_VOICES if yandex_ready() else EDGE_VOICES
    cur = get_voice(uid)
    rows = []
    for i in range(0, len(voices), 2):
        row = []
        for v in voices[i:i + 2]:
            row.append({"text": ("✅ " if v["id"] == cur else "")
                        + v["label"].split(" — ")[0],
                        "callback_data": f"set:voice:{v['id']}"})
        rows.append(row)
    rows.append([{"text": "← Назад", "callback_data": "menu:settings"}])
    return {"inline_keyboard": rows}


def speed_menu_keyboard(uid: int) -> dict:
    cur = get_speed(uid)
    speeds = [0.8, 0.9, 0.95, 1.0, 1.1, 1.2]
    rows = []
    for i in range(0, len(speeds), 3):
        rows.append([{"text": ("✅ " if abs(cur - s) < 0.001 else "")
                      + f"{s:.2f}×", "callback_data": f"set:speed:{s}"}
                     for s in speeds[i:i + 3]])
    rows.append([{"text": "← Назад", "callback_data": "menu:settings"}])
    return {"inline_keyboard": rows}


def data_menu_keyboard(uid: int) -> dict:
    cur = set(get_scopes(uid))
    rows = []
    for key, label in SCOPE_NAMES.items():
        rows.append([{"text": ("✅ " if key in cur else "▫️ ") + label,
                      "callback_data": f"set:scope:{key}"}])
    rows.append([{"text": "← Назад", "callback_data": "menu:settings"}])
    return {"inline_keyboard": rows}


def handle_settings(chat_id: int, uid: int, message_id: int = None,
                    text: str = "⚙️ Настройки ответа") -> None:
    if message_id:
        edit_message(chat_id, message_id, text, settings_keyboard(uid))
    else:
        send_message(chat_id, text, settings_keyboard(uid))


def tg_upload(method: str, chat_id: int, field: str, filename: str,
              data: bytes, mime: str, caption: str = None) -> dict:
    """multipart-загрузка файла в Bot API (sendVoice/sendAudio)."""
    boundary = "----BORAG" + str(int(time.time() * 1000))

    def part(name, value):
        return (f"--{boundary}\r\nContent-Disposition: form-data; "
                f"name=\"{name}\"\r\n\r\n{value}\r\n").encode()

    body = part("chat_id", str(chat_id))
    if caption:
        body += part("caption", caption)
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f"name=\"{field}\"; filename=\"{filename}\"\r\n"
             f"Content-Type: {mime}\r\n\r\n").encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    for attempt in range(3):
        req = urllib.request.Request(
            f"{API}/{method}", data=body,
            headers={"Content-Type":
                     f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                res = json.loads(r.read())
            if not res.get("ok"):
                raise RuntimeError(f"TG {method}: {str(res)[:200]}")
            return res["result"]
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < 2:
                time.sleep(1 + attempt * 2)
                continue
            raise
    raise RuntimeError("upload failed")


def tts_telegram(text: str, voice: str, speed: float):
    if yandex_ready():
        try:
            return (yandex_tts(text, voice, speed, fmt="oggopus"),
                    "voice.ogg", "audio/ogg", True)
        except Exception as e:
            log.warning("yandex voice failed: %s — edge fallback", e)
    return (edge_tts_mp3(text, resolve_voice("male"), speed),
            "voice.mp3", "audio/mpeg", False)


def send_voice_answer(chat_id: int, text: str, voice: str, speed: float,
                      caption: str = "🔊 Озвучка ответа") -> None:
    try:
        data, fname, mime, is_voice = tts_telegram(text, voice, speed)
        if is_voice:
            tg_upload("sendVoice", chat_id, "voice", fname, data, mime, caption)
        else:
            tg_upload("sendAudio", chat_id, "audio", fname, data, mime, caption)
    except Exception as e:
        log.warning("send voice failed: %s", e)
        try:
            send_message(chat_id, f"Не удалось озвучить: {str(e)[:150]}")
        except Exception:
            pass


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
    """Вызов Bot API. Повторяет 5xx/сетевые ошибки, бросает RuntimeError."""
    body = json.dumps(params).encode()
    last = None
    for attempt in range(3):
        req = urllib.request.Request(
            f"{API}/{method}", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=http_timeout) as r:
                data = json.loads(r.read())
            if not data.get("ok"):
                raise RuntimeError(f"TG {method}: {str(data)[:200]}")
            return data["result"]
        except urllib.error.HTTPError as e:
            last = e
            if e.code >= 500 and attempt < 2:
                time.sleep(1 + attempt * 2)
                continue
            try:
                detail = e.read()[:300]
            except Exception:
                detail = b""
            raise RuntimeError(f"TG {method} HTTP {e.code}: {detail!r}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt < 2:
                time.sleep(1 + attempt * 2)
                continue
            raise
    raise RuntimeError(f"TG {method} failed: {last}")


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


def chat_message_url(chat_id: int, message_id: int, sec: int = None) -> str:
    """Ссылка на сообщение в этом чате (для скопированного аудио).

    Для супергрупп/каналов Telegram использует t.me/c/<id без -100>/<msg>;
    для приватного чата — тот же формат с id пользователя (проверяем на
    практике; если клиент не откроет — останется ссылка на оригинал).
    """
    cid = str(chat_id)
    if cid.startswith("-100"):
        cid = cid[4:]
    elif cid.startswith("-"):
        cid = cid[1:]
    url = f"https://t.me/c/{cid}/{message_id}"
    if sec and int(sec) > 0:
        url += f"?t={int(sec)}"
    return url


# Карта «дата сохбета → message_id в группе-источнике» (data/sohbet_links.json,
# скачивается fetch_assets.py). Ссылка t.me/c/… открывается только
# у участников группы — это и нужно для проверки ответов.
LINKS_PATH = BOT_DIR / "data" / "sohbet_links.json"
PRACTICE_AUDIO_DIR = Path(os.environ.get(
    "BO_PRACTICE_AUDIO_DIR", str(BOT_DIR / "data" / "practice_audio")))


def practice_links() -> dict:
    """Карта practice-источник → сообщение с аудио (practice_links.json)."""
    try:
        from rag_core import practice_links as _pl
        return _pl()
    except Exception:
        return {}


def practice_by_mid(mid: int):
    for sid, v in practice_links().items():
        if int(v.get("m") or 0) == int(mid):
            return sid, v
        for p in (v.get("parts") or []):
            if int(p.get("m") or 0) == int(mid):
                return sid, p
    return None, None


def practice_audio_path(mid: int):
    rec = practice_by_mid(mid)[1]
    if not rec:
        return None
    ext = Path(rec.get("file") or "x.mp3").suffix or ".mp3"
    p = PRACTICE_AUDIO_DIR / f"{int(mid)}{ext}"
    return p if p.exists() else None
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


def set_audio_caption_link(chat_id: int, message_id: int, sec: int) -> bool:
    """Делает подпись к аудио кликабельным таймкодом на это же сообщение."""
    url = chat_message_url(chat_id, message_id, sec)
    try:
        tg("editMessageCaption", chat_id=chat_id, message_id=message_id,
           caption=f'▶️ <a href="{url}">{fmt_ts(sec)}</a>',
           parse_mode="HTML")
        return True
    except Exception as e:
        log.warning("caption link failed: %s", e)
        return False


def send_top_source_audio(chat_id: int, res: dict) -> None:
    """Одно сообщение: скопированное аудио с подписью-таймкодом.
    Никаких вторых reply-сообщений со ссылками на группу."""
    if not SOURCE_CHAT_ID:
        return
    for c in res.get("citations", []):
        mid = c.get("audio_mid")
        if mid:
            try:
                title = c.get("title") or "Практика"
                tg("copyMessage", chat_id=chat_id,
                   from_chat_id=int(SOURCE_CHAT_ID),
                   message_id=int(mid),
                   caption=f"▶️ {title}")
            except Exception as e:
                log.warning("practice audio copy failed mid=%s: %s", mid, e)
            return
        ref = source_audio_ref(c["sohbet_date"], c.get("sec"))
        if not ref:
            continue
        msg_id, sec = ref
        ts = fmt_ts(sec)
        tg("copyMessage", chat_id=chat_id,
           from_chat_id=int(SOURCE_CHAT_ID), message_id=msg_id,
           caption=(f"▶️ Основной источник: сохбет "
                    f"{c['sohbet_date']}, слушайте с {ts}"))
        return


def handle_play_practice(chat_id: int, payload: str) -> None:
    """Deep link из веба: /start playp_<msg_id> → копирует аудио практики."""
    if not SOURCE_CHAT_ID:
        send_message(chat_id, "Аудио-источник не настроен (BO_SOURCE_CHAT_ID).")
        return
    try:
        mid = int(payload.split("_")[0])
    except Exception:
        send_message(chat_id, "Формат: /start playp_266")
        return
    sid, rec = practice_by_mid(mid)
    if not rec:
        send_message(chat_id, f"Не нашёл аудио практики для сообщения {mid}.")
        return
    title = "Практика"
    try:
        from rag_core import practice_ref
        p = practice_ref(sid) if sid else None
        if p and p.get("title"):
            title = p["title"]
    except Exception:
        pass
    tg("copyMessage", chat_id=chat_id,
       from_chat_id=int(SOURCE_CHAT_ID), message_id=mid,
       caption=f"▶️ {title}")


def handle_play(chat_id: int, payload: str) -> None:
    """Deep link из веба: /start play_<date>_<sec> → копирует аудио в чат
    и ставит кликабельный таймкод в подпись (одно сообщение)."""
    if not SOURCE_CHAT_ID:
        send_message(chat_id, "Аудио-источник не настроен (BO_SOURCE_CHAT_ID).")
        return
    if "_" not in payload:
        send_message(chat_id, "Формат: /start play_2026-05-31_123")
        return
    date, sec_str = payload.rsplit("_", 1)
    try:
        sec = int(sec_str)
    except Exception:
        sec = 0
    ref = source_audio_ref(date, sec)
    if not ref:
        send_message(chat_id, f"Не нашёл аудио для {date}. "
                              "Возможно, это практика без карты сообщений.")
        return
    msg_id, offset = ref
    ts = fmt_ts(offset)
    tg("copyMessage", chat_id=chat_id,
       from_chat_id=int(SOURCE_CHAT_ID), message_id=msg_id,
       caption=(f"▶️ Основной источник: сохбет {date}, "
                f"слушайте с {ts}"))


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
    return {"inline_keyboard": [[btn("zhivoy")], [btn("publication")],
                                [{"text": "⚙️ Настройки",
                                  "callback_data": "menu:settings"}]]}


def get_mode(user_id: int) -> str:
    return user_mode.get(user_id, "zhivoy")


def format_answer(res: dict) -> str:
    text = res["answer"]
    if res["citations"]:
        text += "\n\n— Источники:"
        for c in res["citations"]:
            if c.get("source_type") in ("method", "theory", "reference"):
                label = c.get("title") or "Источник"
                if c.get("source_type") == "reference" and c.get("author"):
                    label += f" — {c['author']}"
            else:
                label = f"сохбет {c['sohbet_date']}"
            if c.get("source_type") != "method":
                sec = c.get("sec")
                url = (f"https://t.me/alexenur_bo_bot?start=play_"
                       f"{c['sohbet_date']}_{sec}"
                       if sec is not None else None)
            else:
                mid = c.get("audio_mid")
                url = (f"https://t.me/alexenur_bo_bot?start=playp_{mid}"
                       if mid else None)
            if url:
                label = f'<a href="{url}">{label}</a>'
            line = f"\n• {label}"
            if (c.get("source_type") not in ("method", "theory", "reference")
                    and c.get("time")):
                ts = c["time"]
                line += (f', <a href="{url}">{ts}</a>' if url
                         else f", {ts}")
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

    status_id = send_message(
        chat_id, "⏳ …",
        markup={"inline_keyboard": [[{
            "text": "⏹ Стоп", "callback_data": f"cancel:{user_id}"}]]})
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

    cancel_ev = threading.Event()
    cancel_events[user_id] = cancel_ev
    try:
        from rag_core import answer
        res = answer(question, mode, progress=progress, cancel=cancel_ev,
                     scopes=get_scopes(user_id))
    except Exception as e:
        cancel_events.pop(user_id, None)
        log.exception("rag error")
        delete_message(chat_id, status_id)
        send_message(chat_id, "Произошла ошибка при поиске ответа. "
                              f"Попробуй ещё раз чуть позже.\n({e})")
        return
    cancel_events.pop(user_id, None)
    if res.get("cancelled"):
        edit_message(chat_id, status_id, "⏹ Остановлено.",
                     {"inline_keyboard": []})
        return
    delete_message(chat_id, status_id)
    send_long(chat_id, format_answer(res))
    if get_tts(user_id):
        send_voice_answer(chat_id, res["answer"], get_voice(user_id),
                          get_speed(user_id))
    else:
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
        uid = cq["from"]["id"]
        msg = cq.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        mid = msg.get("message_id")

        def ack(text=None):
            try:
                params = {"callback_query_id": cq["id"]}
                if text:
                    params["text"] = text
                tg("answerCallbackQuery", **params)
            except Exception:
                pass

        if data.startswith("cancel:"):
            try:
                cuid = int(data.split(":", 1)[1])
            except Exception:
                cuid = None
            ev = cancel_events.get(cuid)
            if ev:
                ev.set()
            if mid and chat_id:
                edit_message(chat_id, mid, "⏹ Останавливаю…",
                             {"inline_keyboard": []})
            ack("Останавливаю…")
            return
        if data == "menu:settings":
            if chat_id:
                handle_settings(chat_id, uid, mid)
            ack()
            return
        if data == "menu:voice":
            if chat_id:
                edit_message(chat_id, mid, "🎤 Выберите голос:",
                             voice_menu_keyboard(uid))
            ack()
            return
        if data == "menu:speed":
            if chat_id:
                edit_message(chat_id, mid, "⚡ Выберите скорость:",
                             speed_menu_keyboard(uid))
            ack()
            return
        if data == "menu:data":
            if chat_id:
                edit_message(chat_id, mid,
                             "📚 Какие данные использовать?\n"
                             "Можно включить несколько. По умолчанию — "
                             "только сохбеты.",
                             data_menu_keyboard(uid))
            ack()
            return
        if data.startswith("set:scope:"):
            scope = data.split(":", 2)[2]
            if scope in SCOPE_NAMES:
                cur = set(user_scopes.get(uid) or {"sohbets"})
                if scope in cur:
                    cur.discard(scope)
                else:
                    cur.add(scope)
                if not cur:
                    cur = {"sohbets"}
                user_scopes[uid] = cur
                save_settings()
            if chat_id:
                edit_message(chat_id, mid,
                             "📚 Какие данные использовать?\n"
                             "Можно включить несколько. По умолчанию — "
                             "только сохбеты.",
                             data_menu_keyboard(uid))
            ack()
            return
        if data.startswith("set:mode:"):
            mode = data.split(":", 2)[2]
            if mode in MODE_NAMES:
                user_mode[uid] = mode
                save_settings()
            if chat_id:
                handle_settings(chat_id, uid, mid)
            ack()
            return
        if data.startswith("set:voice:"):
            user_voice[uid] = data.split(":", 2)[2]
            save_settings()
            if chat_id:
                edit_message(chat_id, mid, "🎤 Выберите голос:",
                             voice_menu_keyboard(uid))
            ack()
            return
        if data.startswith("set:speed:"):
            try:
                sp = float(data.split(":", 2)[2])
            except Exception:
                sp = 0.95
            user_speed[uid] = max(0.5, min(1.5, sp))
            save_settings()
            if chat_id:
                edit_message(chat_id, mid, "⚡ Выберите скорость:",
                             speed_menu_keyboard(uid))
            ack()
            return
        if data.startswith("set:tts:"):
            user_tts[uid] = data.endswith(":on")
            save_settings()
            if chat_id:
                handle_settings(chat_id, uid, mid)
            ack()
            return
        if data == "test:voice":
            ack("Готовлю голос…")
            send_voice_answer(chat_id or uid,
                              "Ассаляму алейкум. Это проверка выбранного "
                              "голоса и скорости.", get_voice(uid),
                              get_speed(uid))
            return
        if data.startswith("mode:"):
            mode = data.split(":", 1)[1]
            user_mode[uid] = mode
            save_settings()
            if chat_id:
                edit_message(chat_id, mid, f"Режим: {MODE_NAMES[mode]}",
                             mode_keyboard(mode))
        ack()
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

    if text.startswith("/start playp_"):
        try:
            handle_play_practice(chat_id, text[len("/start playp_"):])
        except Exception as e:
            log.exception("practice play deep link error")
            send_message(chat_id, f"Не удалось открыть аудио: {e}")
        return
    if text.startswith("/start play_"):
        try:
            handle_play(chat_id, text[len("/start play_"):])
        except Exception as e:
            log.exception("play deep link error")
            send_message(chat_id, f"Не удалось открыть аудио: {e}")
        return
    if text.startswith("/start"):
        send_message(chat_id,
                     "Ассаляму алейкум. Я — цифровой двойник Бориса Олеговича, "
                     "отвечаю по записям его сохбетов.\n\n"
                     "Просто напиши вопрос. Настройки голоса и скорости — "
                     "кнопка «⚙️ Настройки» или /settings.",
                     mode_keyboard(get_mode(user_id)))
    elif text.startswith("/mode"):
        send_message(chat_id,
                     f"Текущий режим: {MODE_NAMES[get_mode(user_id)]}",
                     mode_keyboard(get_mode(user_id)))
    elif text.startswith("/settings"):
        handle_settings(chat_id, user_id)
    elif text.startswith("/voice"):
        send_message(chat_id, "🎤 Выберите голос:",
                     voice_menu_keyboard(user_id))
    elif text.startswith("/speed"):
        send_message(chat_id, "⚡ Выберите скорость:",
                     speed_menu_keyboard(user_id))
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


def _run_update(upd: dict) -> None:
    """Обработка обновления в отдельном потоке (webhook уже ответил 200).

    Callback-кнопки (в т.ч. «Стоп») обрабатываются вне _tg_lock, чтобы
    не ждать завершения текущего ответа.
    """
    if "callback_query" in upd:
        try:
            handle_update(upd)
        except Exception:
            log.exception("callback handler error")
        return
    with _tg_lock:
        try:
            handle_update(upd)
        except Exception:
            log.exception("update handler error")


def set_webhook() -> None:
    url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not url:
        log.warning("RENDER_EXTERNAL_URL не задан — webhook не установлен")
        return
    hook = f"{url}/tg/{WEBHOOK_SECRET}"
    try:
        tg("setWebhook", url=hook, secret_token=WEBHOOK_SECRET,
           allowed_updates=["message", "callback_query"],
           drop_pending_updates=False)
        log.info("webhook set: %s", hook)
    except Exception as e:
        log.error("setWebhook failed: %s", e)


def polling_loop() -> None:
    offset = 0
    try:
        tg("deleteWebhook", drop_pending_updates=False)
    except Exception:
        pass
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

    def _send_file(self, p: Path, ctype: str):
        """Отдаёт файл с поддержкой Range (перемотка в <audio>)."""
        total = p.stat().st_size
        rng = self.headers.get("Range") or ""
        start, end = 0, total - 1
        code = 200
        if rng.startswith("bytes="):
            try:
                spec = rng[len("bytes="):].split(",")[0]
                a, b = spec.split("-", 1)
                if a:
                    start = int(a)
                    end = int(b) if b else total - 1
                else:
                    start = max(0, total - int(b))
                start = max(0, min(start, total - 1))
                end = max(start, min(end, total - 1))
                code = 206
            except Exception:
                self._send(416, b"", "text/plain")
                return
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, max-age=3600")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(262144, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                left -= len(chunk)

    def _audio_allowed(self) -> bool:
        if not WEB_PASSWORD:
            return True
        q = urllib.parse.parse_qs(
            urllib.parse.urlparse(self.path).query)
        code = (q.get("code") or [""])[0] or self.headers.get(
            "X-Access-Code", "")
        return code == WEB_PASSWORD

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
        elif path == "/tts/info":
            provider = "yandex" if yandex_ready() else "edge"
            voices = YANDEX_VOICES if provider == "yandex" else EDGE_VOICES
            default = resolve_voice(YC_VOICE if provider == "yandex"
                                    else EDGE_VOICE)
            self._send(200, json.dumps({
                "provider": provider,
                "voices": voices,
                "default": default,
                "speed": 0.95,
            }, ensure_ascii=False))
        elif path.startswith("/practice/"):
            if not self._audio_allowed():
                self._send(401, json.dumps(
                    {"error": "auth", "message": "Нужен код доступа."},
                    ensure_ascii=False))
                return
            mid_s = path.rsplit("/", 1)[-1].split(".")[0]
            try:
                mid = int(mid_s)
            except Exception:
                self._send(404, json.dumps({"error": "not found"}))
                return
            p = practice_audio_path(mid)
            if not p:
                self._send(404, json.dumps(
                    {"error": "no audio", "mid": mid}, ensure_ascii=False))
                return
            ctype = ("audio/wav" if p.suffix.lower() == ".wav"
                     else "audio/mpeg")
            self._send_file(p, ctype)
        else:
            self._send(404, json.dumps({"error": "not found"}))

    do_HEAD = do_GET

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        # Telegram webhook: отвечаем 200 сразу, обработку — в фоне
        if USE_WEBHOOK and path == f"/tg/{WEBHOOK_SECRET}":
            if (self.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
                    != WEBHOOK_SECRET):
                self._send(403, json.dumps({"error": "forbidden"}))
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                upd = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._send(400, json.dumps({"error": "bad update"}))
                return
            threading.Thread(target=_run_update, args=(upd,),
                             daemon=True).start()
            self._send(200, json.dumps({"ok": True}))
            return

        if path == "/tts":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
                text = (payload.get("text") or "").strip()[:1000]
                if not text:
                    raise ValueError("empty")
                voice = resolve_voice(payload.get("voice"))
                try:
                    speed = float(payload.get("speed", 0.95))
                except Exception:
                    speed = 0.95
                speed = max(0.5, min(1.5, speed))
                if yandex_ready():
                    try:
                        audio = yandex_tts(text, voice, speed)
                        log.info("tts yandex voice=%s speed=%.2f len=%d",
                                 voice, speed, len(audio))
                    except Exception as e:
                        log.warning("yandex tts failed: %s — edge fallback", e)
                        audio = edge_tts_mp3(text, resolve_voice("male"), speed)
                else:
                    audio = edge_tts_mp3(text, voice, speed)
                    log.info("tts edge voice=%s speed=%.2f len=%d",
                             voice, speed, len(audio))
                self._send(200, audio, "audio/mpeg")
            except Exception as e:
                log.warning("tts failed: %s", e)
                self._send(503, json.dumps(
                    {"error": "tts",
                     "message": "Серверная озвучка недоступна — включите голос браузера."},
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
            scopes = payload.get("scopes") or ["sohbets"]
            if isinstance(scopes, str):
                scopes = [scopes]
            scopes = [s for s in scopes
                      if s in ("sohbets", "theory", "literature")] or ["sohbets"]
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
            res = answer(question, mode, scopes=scopes)
            body = {
                "answer": res["answer"],
                "citations": res["citations"],
                "mode": res["mode"],
                "scopes": scopes,
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
        if USE_WEBHOOK:
            try:
                info = tg("getWebhookInfo", http_timeout=20)
                if not info.get("url"):
                    log.warning("webhook потерян — восстанавливаю")
                    set_webhook()
            except Exception as e:
                log.warning("webhook check failed: %s", e)


def main() -> None:
    load_settings()
    start_http_server()  # порт сразу — иначе Render убьёт по таймауту
    threading.Thread(target=self_ping_loop, daemon=True).start()
    if os.environ.get("BO_WEB_ONLY") == "1":
        log.info("BO_WEB_ONLY=1 — Telegram polling выключен")
        while True:
            time.sleep(3600)
    if not TOKEN:
        sys.exit("Нет токена: env BO_BOT_TOKEN")
    if USE_WEBHOOK:
        set_webhook()
        log.info("webhook mode — polling не запускаем")
        while True:
            time.sleep(3600)
    polling_loop()


if __name__ == "__main__":
    main()
