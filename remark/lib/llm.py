"""Generic OpenAI-compatible chat client with a resumable JSONL cache.

Environment:
    LLM_API_KEY    API key
    LLM_ENDPOINT   base URL of an OpenAI-compatible chat-completions API
    LLM_MODEL      default model name (override per request with "model")
    GEMINI_API_KEY key for requests whose model name starts with "gemini" (REST generateContent)

Usage:
    from lib.llm import chat, batch_chat
    text = chat([{"role": "user", "content": "..."}])
    results = batch_chat(requests, cache_path="remark/outputs/x.jsonl")
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
_client: OpenAI | None = None
_client_lock = threading.Lock()


def get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ["LLM_ENDPOINT"],
                timeout=180,
                max_retries=0)
        return _client


def default_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-5.6")


def chat(
    messages: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 1024,
    retries: int = 6,
    **kwargs: Any) -> dict[str, Any]:
    """One call with exponential backoff. Returns {"text", "usage"}.

    429 rate-limit errors are retried patiently (up to `rate_limit_retries`) and do not
    count against `retries`.
    """
    client = get_client()
    delay = 2.0
    failures = 0
    rate_limited = 0
    rate_limit_retries = 200
    while True:
        try:
            t_call = time.time()
            resp = client.chat.completions.create(
                model=model or default_model(),
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
                **kwargs)
            usage = resp.usage.model_dump() if resp.usage else {}
            return {"text": resp.choices[0].message.content or "", "usage": usage,
                    "latency_s": round(time.time() - t_call, 2), "rate_limited": rate_limited}
        except Exception as exc:  # noqa: BLE001 - retry any transport/API error
            if "429" in repr(exc):
                rate_limited += 1
                if rate_limited >= rate_limit_retries:
                    raise
                time.sleep(8.0 + random.random() * 12.0)
                continue
            failures += 1
            if failures >= retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def gemini_chat(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int = 1024,
    retries: int = 6) -> dict[str, Any]:
    """One Gemini call (REST generateContent, key GEMINI_API_KEY sent as a header), default sampling settings.

    Same return shape as chat(). Thought parts are dropped from the text; an empty answer raises so the
    request stays retryable in batch_chat. 429/5xx are retried patiently.
    """
    import urllib.error
    import urllib.request

    key = os.environ["GEMINI_API_KEY"]
    role = {"user": "user", "assistant": "model"}
    body = {"contents": [{"role": role[m["role"]], "parts": [{"text": m["content"]}]} for m in messages],
            "generationConfig": {"maxOutputTokens": max_tokens}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    delay, failures, rate_limited = 2.0, 0, 0
    while True:
        try:
            t_call = time.time()
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})
            d = json.load(urllib.request.urlopen(req, timeout=600))
            cand = (d.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
            if not text.strip():
                raise RuntimeError(f"empty answer, finishReason={cand.get('finishReason')}")
            u = d.get("usageMetadata", {})
            usage = {"prompt_tokens": u.get("promptTokenCount", 0), "completion_tokens": u.get("candidatesTokenCount", 0),
                     "thoughts_tokens": u.get("thoughtsTokenCount", 0), "model_version": d.get("modelVersion")}
            return {"text": text, "usage": usage, "latency_s": round(time.time() - t_call, 2), "rate_limited": rate_limited}
        except Exception as exc:  # noqa: BLE001 - retry any transport/API error
            code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
            if code == 429 or (code is not None and code >= 500):
                rate_limited += 1
                if rate_limited >= 200:
                    raise RuntimeError(f"HTTP {code} after {rate_limited} retries") from None
                time.sleep(8.0 + random.random() * 12.0)
                continue
            failures += 1
            if failures >= retries:
                raise RuntimeError(repr(exc).replace(key, "<key>")[:300]) from None
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def request_key(req: dict[str, Any]) -> str:
    payload = json.dumps(
        {"messages": req["messages"], "model": req.get("model"), "max_tokens": req.get("max_tokens")},
        sort_keys=True,
        ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def batch_chat(
    requests: Iterable[dict[str, Any]],
    cache_path: str | Path,
    workers: int = 32) -> dict[str, dict[str, Any]]:
    """Run requests concurrently with a resumable JSONL cache.

    Each request: {"id": str, "messages": [...], optional "model", "max_tokens", "meta"}.
    Returns {id: {"text", "usage", "meta"}}. Already-cached ids are skipped.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in rec:
                    done[rec["id"]] = rec
    todo = [r for r in requests if r["id"] not in done]
    write_lock = threading.Lock()

    def run(req: dict[str, Any]) -> dict[str, Any]:
        try:
            m = req.get("model") or ""
            fn = gemini_chat if m.startswith("gemini") else chat
            out = fn(req["messages"], model=req.get("model"), max_tokens=req.get("max_tokens", 1024))
            rec = {"id": req["id"], "text": out["text"], "usage": out["usage"], "meta": req.get("meta"),
                   "latency_s": out.get("latency_s"), "rate_limited": out.get("rate_limited")}
        except Exception as exc:  # noqa: BLE001
            rec = {"id": req["id"], "error": repr(exc)[:500], "meta": req.get("meta")}
        with write_lock, cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(run, r) for r in todo]):
            rec = fut.result()
            if "error" not in rec:
                done[rec["id"]] = rec
    return done
