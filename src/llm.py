"""Groq chat client with JSON mode, rate-limit backoff, and a content-addressed disk cache.

The free Groq tier allows ~8K tokens/minute per model, so every agent step is a single
compact call and identical prompts are never paid for twice (re-running the eval after a
change only spends tokens on the steps whose inputs actually changed).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import settings


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)


USAGE = Usage()
_lock = threading.Lock()


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


class LLM:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1200, cache: bool = True) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not settings.groq_api_key:
                raise LLMError("GROQ_API_KEY is not set; copy .env.example to .env and add a key.")
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1", max_retries=0)
        return self._client

    def _extra(self) -> dict[str, Any]:
        # Keep hidden reasoning short: it counts against the per-minute token budget.
        if self.model.startswith("openai/gpt-oss"):
            return {"reasoning_effort": "low"}
        if self.model.startswith("qwen/"):
            return {"reasoning_effort": "none"}
        return {}

    def _cache_path(self, payload: dict[str, Any]):
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return settings.llm_cache_dir / key[:2] / f"{key}.json"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "json": json_mode, "system": system, "user": user}
        path = self._cache_path(payload)
        if self.cache and path.exists():
            USAGE.cached += 1
            return json.loads(path.read_text(encoding="utf-8"))["content"]

        kwargs: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        for attempt in range(8):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    extra_body=self._extra(), **kwargs,
                )
                break
            except Exception as exc:  # openai.RateLimitError / APIStatusError / connection errors
                status = getattr(exc, "status_code", None)
                message = str(exc)
                if status == 429 and "per day" in message.lower():
                    raise LLMError(f"Daily Groq token quota exhausted for {self.model}: {message}") from exc
                if status not in (429, 500, 502, 503) and "json_validate_failed" not in message and "Connection" not in type(exc).__name__:
                    raise LLMError(f"{self.model}: {message}") from exc
                wait = 2 ** min(attempt, 5)
                match = re.search(r"try again in ([\d.]+)(ms|s)", message)
                if match:
                    wait = float(match.group(1)) / (1000 if match.group(2) == "ms" else 1) + 0.5
                time.sleep(wait)
        else:
            raise LLMError(f"{self.model}: gave up after repeated rate-limit/server errors")

        content = response.choices[0].message.content or ""
        with _lock:
            USAGE.calls += 1
            if response.usage:
                USAGE.prompt_tokens += response.usage.prompt_tokens
                USAGE.completion_tokens += response.usage.completion_tokens
                USAGE.by_model[self.model] = USAGE.by_model.get(self.model, 0) + response.usage.total_tokens
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"content": content}), encoding="utf-8")
        return content

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        text = self.complete(system, user, json_mode=True)
        try:
            return _extract_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"{self.model} returned invalid JSON: {text[:200]}") from exc


def generator() -> LLM:
    return LLM(settings.generator_model, max_tokens=1500)


def agent() -> LLM:
    return LLM(settings.agent_model, max_tokens=1200)


def judge() -> LLM:
    return LLM(settings.judge_model, max_tokens=1500)
