"""Minimal, swappable Groq generation interface for RAG answers and evaluation."""

from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv


class GroqGenerator:
    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_GENERATION_MODEL")
        if not self.api_key or not self.model:
            raise ValueError("Set GROQ_API_KEY and GROQ_GENERATION_MODEL in .env.")

    def answer(self, question: str, contexts: Sequence[str]) -> str:
        from openai import OpenAI

        context_block = "\n\n".join(f"[Context {index}] {text}" for index, text in enumerate(contexts, start=1))
        response = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1").chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "system", "content": "Answer only from the supplied SEC filing context. If the context is insufficient, say so. Keep the answer concise."}, {"role": "user", "content": f"Question: {question}\n\nSEC filing context:\n{context_block}"}],
        )
        return response.choices[0].message.content or ""
