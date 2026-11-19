#!/usr/bin/env python3
"""go_llm_call.py — минимальный вызов LLM через OpenCode Go API.

Назначение: замена rate-limited z-ai CLI (~30 вызовов/час, баны часами)
в batch-постобработке (clean_and_curate, summarize). Лимиты Go на порядки
выше (DeepSeek V4.1 Flash: ~34,7 тыс. запросов/5ч на 11.2026).

Использование:
    python3 scripts/go_llm_call.py --model "deepseek-v4.1-flash" \
        --prompt-file prompt.txt [--system "..."] [--max-tokens 4000]

Ключ: credentials/opencode_go.key или env OPENCODE_GO_KEY.
Wire-протокол выбирается по имени модели (можно переопределить --protocol):
    openai    — GLM, Kimi, DeepSeek, MiMo, LongCat, Hy, Space Bunny, previews
    anthropic — Claude Haiku, MiniMax, Qwen3.8/3.7
    responses — Grok, GPT Luna, Muse Spark (пока НЕ реализовано: exit 4)

При 429 печатает "RATE_LIMITED" в stdout и завершается с кодом 3.
Ретраи внутри скрипта ОТСУТСТВУЮТ намеренно (ретраи продлевают баны —
паттерн z-ai; внешний цикл решает сам). Прогресс и возобновление — забота
вызывающего скрипта.

ПЕРЕД ПЕРВЫМ ИСПОЛЬЗОВАНИЕМ: проверь на одном вызове (правило проекта —
сначала 1 файл, потом batch). Список живых моделей: scripts/go_models.sh.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINTS = {
    "openai": "https://opencode.ai/inference/go/openai/v1/chat/completions",
    "anthropic": "https://opencode.ai/inference/go/anthropic/v1/messages",
    "responses": "https://opencode.ai/inference/go/openai/v1/responses",
}

# Префиксы имён моделей -> wire-протокол (состояние на 2026-11; сверяйся с
# выводом scripts/go_models.sh — ростер меняется)
PROTOCOL_BY_NAME = [
    (("claude", "minimax", "qwen3.8", "qwen3.7"), "anthropic"),
    (("grok", "gpt", "muse"), "responses"),
]


def detect_protocol(model: str) -> str:
    m = model.lower()
    for prefixes, proto in PROTOCOL_BY_NAME:
        if any(p in m for p in prefixes):
            return proto
    return "openai"


def load_key() -> str:
    key = os.environ.get("OPENCODE_GO_KEY", "").strip()
    if key:
        return key
    path = os.path.join(os.path.dirname(__file__), "..", "credentials", "opencode_go.key")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    sys.exit("Нет ключа: credentials/opencode_go.key или env OPENCODE_GO_KEY")


class RateLimited(Exception):
    """429 от Go API (в т.ч. для in-process вызовов из rag_core)."""


def call_model(model: str, prompt: str, system: str = None,
               max_tokens: int = 4000, protocol: str = None) -> str:
    """Один вызов LLM. Бросает RateLimited при 429, RuntimeError при прочем."""
    proto = protocol or detect_protocol(model)
    if proto == "responses":
        raise RuntimeError("Протокол responses (Grok/GPT Luna/Muse) не "
                           "реализован — выбери модель openai/anthropic")
    url = ENDPOINTS[proto]
    key = load_key()

    if proto == "openai":
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    else:  # anthropic messages
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               # UA по умолчанию (Python-urllib) Cloudflare режет с 403/1010
               "User-Agent": "curl/8.0 (transcribe curator)"}
    if proto == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    # Go API требует session id для роутинга (без него MissingSessionID)
    headers["x-opencode-session"] = os.environ.get(
        "OPENCODE_GO_SESSION", f"curator-{os.path.basename(os.getcwd())}")

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited(f"429 на {model}")
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:500]!r}")

    if proto == "openai":
        msg = data["choices"][0]["message"]
        # reasoning-модели (deepseek) могут вернуть пустой content при
        # нехватке max_tokens — reasoning_tokens съедают лимит
        if not msg.get("content") and msg.get("reasoning_content"):
            raise RuntimeError("EMPTY_CONTENT: модель ушла в reasoning, "
                               "увеличь max_tokens")
        return msg["content"].strip()
    return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="точный ID модели из /v1/models")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--protocol", choices=list(ENDPOINTS), default=None)
    args = ap.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
    if not prompt:
        sys.exit("Нужен --prompt или --prompt-file")

    try:
        print(call_model(args.model, prompt, system=args.system,
                         max_tokens=args.max_tokens, protocol=args.protocol))
    except RateLimited:
        print("RATE_LIMITED")
        return 3
    except RuntimeError as e:
        sys.exit(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
