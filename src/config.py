"""Central configuration, read once from the environment (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = ROOT / "data" / "raw_filings"
    corpus_path: Path = ROOT / "data" / "corpus" / "chunks.jsonl"
    qdrant_path: Path = ROOT / "data" / "index" / "qdrant"
    qdrant_url: str | None = os.getenv("QDRANT_URL") or None
    collection: str = "findocgpt_chunks"
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-base")
    llm_cache_dir: Path = ROOT / "data" / "llm_cache"

    # Three Groq models with separate quotas: answers, fast agent steps, and the eval judge.
    # The judge is deliberately a different model family from the generator to limit self-preference bias.
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    generator_model: str = _env("GENERATOR_MODEL", "openai/gpt-oss-120b")
    agent_model: str = _env("AGENT_MODEL", "openai/gpt-oss-20b")
    judge_model: str = _env("JUDGE_MODEL", "qwen/qwen3.8-27b")

    candidates_per_query: int = 20
    rerank_keep: int = 6
    max_context_chunks: int = 8
    retry_threshold: float = field(default=float(_env("RETRY_THRESHOLD", "0.2")))


settings = Settings()
