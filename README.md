# bo_rag_bot — цифровой двойник БО (Telegram + веб-чат)

RAG-бот по корпусу сохбетов БО. Ищет по 59 тыс. фрагментов и 6,5 тыс. пар
вопрос-ответ, отвечает строго по найденным цитатам, показывает дату и таймкод.

## Что умеет

- **Telegram-бот** (`@alexenur_bo_bot`): webhook (secret token) или
  long polling, режимы «Живой» / «Для публикации», источники со ссылками
  на аудио в группе.
- **Веб-чат** на том же сервисе: `GET /` — страница чата, `POST /ask` — JSON.
- **Голос**:
  - ввод — Web Speech API в браузере (Chrome/Edge/Safari, ru-RU);
  - озвучка — Yandex SpeechKit, если заданы `YC_API_KEY` + `YC_FOLDER_ID`,
    иначе Edge TTS (русские нейроголоса Microsoft), иначе голос браузера.
- **Ограничение доступа** к Telegram: allowlist по username/id.
- **Лимит веб-чата**: N вопросов в сутки на IP (по умолчанию 40).
- Дисклеймер: ответы по сохбетам, не терапия и не медицинская рекомендация.

## HTTP

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/` | страница веб-чата |
| GET | `/health` | health-check для Render |
| GET | `/stats` | размеры индекса |
| POST | `/ask` | `{"question": "...", "mode": "publication"}` → ответ + цитаты |
| POST | `/tts` | `{"text": "..."}` → mp3 (Yandex SpeechKit) |

`/ask` и `/tts` возвращают `401`, если задан `BO_WEB_PASSWORD` и не передан
заголовок `X-Access-Code`.

## Переменные окружения

| Переменная | Обязательно | Назначение |
|---|---|---|
| `BO_BOT_TOKEN` | да (TG) | токен `@alexenur_bo_bot` |
| `OPENCODE_GO_KEY` | да | ключ LLM (OpenCode Go API) |
| `GH_PAT` | да (build) | чтение приватного релиза с ассетами |
| `BO_ALLOWED_USERNAMES` | для прода | `alexenur,tgnamel` |
| `BO_ALLOWED_IDS` | опционально | числовые TG id |
| `BO_WEB_PASSWORD` | опционально | код доступа к веб-чату |
| `BO_WEB_DAILY_LIMIT` | нет | лимит вопросов на IP в сутки (40) |
| `BO_USE_WEBHOOK` | нет | `1` — Telegram webhook вместо polling |
| `BO_WEBHOOK_SECRET` | для webhook | секрет в URL и заголовке Telegram |
| `BO_GEN_MODEL` | нет | модель LLM (`minimax-m3`) |
| `YC_API_KEY`, `YC_FOLDER_ID` | опционально | Yandex SpeechKit TTS |
| `YC_VOICE` | нет | голос Yandex (`alena` / `filipp`) |
| `BO_EDGE_VOICE` | нет | Edge TTS (`ru-RU-SvetlanaNeural`) |
| `BO_DEBUG` | нет | `1` — диагностика в логах |
| `BO_WEB_ONLY` | нет | `1` — не запускать Telegram polling |

## Данные

На build `fetch_assets.py` скачивает из релиза `bo-rag-data-v1`:

- `data/vectors_par.npy`, `data/vectors_qa.npy` — вектора (384d);
- `data/payloads.db` — тексты и метаданные (SQLite);
- `data/meta.json`, `data/sohbet_links.json`;
- `models/paraphrase-multilingual-MiniLM-L12-...gguf` — эмбеддер llama.cpp.

На runtime в памяти ~300–400 МБ: 59 тыс. векторов + GGUF-модель. Это влезает
в Render free (512 МБ), но с небольшим запасом. При росте корпуса — план 1 ГБ
или внешний Qdrant.

## Render

- Runtime: Python 3
- Root Dir: `bots/bo_rag_bot`
- Build: `pip install -r requirements.txt && python fetch_assets.py && pip install llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl`
- Start: `python bot.py`
- Health check: `/health`
- Free tier засыпает через 15 мин; self-ping каждые 10 мин держит сервис живым.

## Обновление корпуса

В `transcribe`:

```bash
python scripts/release_rag.py --skip-build   # соберёт v11 и загрузит ассет
```

Затем обновить `ASSET_URL` в `fetch_assets.py` на id нового ассета и
передеплоить сервис. База-источник — vault `brain/05_projects/spiritual_books/sohbets/`.
