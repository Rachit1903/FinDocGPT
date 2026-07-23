"""Verify answer claims against retrieved source chunks and remove unsupported text."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from dotenv import load_dotenv

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
TOKEN_PATTERN = re.compile(r"[a-z]+|\d+(?:\.\d+)?", re.I)
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
STOP_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with"}


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    supported: bool
    source_ids: list[str]
    rationale: str


@dataclass(frozen=True)
class VerificationResult:
    verified_answer: str
    verdicts: list[ClaimVerdict]

    @property
    def unsupported_claims(self) -> list[str]:
        return [verdict.claim for verdict in self.verdicts if not verdict.supported]


class ClaimJudge(Protocol):
    def judge(self, claim: str, sources: Sequence[SourceChunk]) -> ClaimVerdict: ...


def extract_claims(answer: str) -> list[str]:
    """Extract atomic returnable claims as individual answer sentences."""
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(answer.strip()) if sentence.strip()]


class LexicalEntailmentJudge:
    """Local, conservative fallback useful for tests and no-network demonstrations."""

    def judge(self, claim: str, sources: Sequence[SourceChunk]) -> ClaimVerdict:
        claim_tokens = {token.lower() for token in TOKEN_PATTERN.findall(claim)} - STOP_WORDS
        claim_numbers = set(NUMBER_PATTERN.findall(claim))
        best_source: SourceChunk | None = None
        best_ratio = 0.0
        for source in sources:
            source_tokens = {token.lower() for token in TOKEN_PATTERN.findall(source.text)}
            if claim_numbers and not claim_numbers.issubset(set(NUMBER_PATTERN.findall(source.text))):
                continue
            ratio = len(claim_tokens & source_tokens) / max(1, len(claim_tokens))
            if ratio > best_ratio:
                best_ratio, best_source = ratio, source
        supported = best_source is not None and best_ratio >= 0.70
        return ClaimVerdict(
            claim=claim, supported=supported, source_ids=[best_source.chunk_id] if supported else [],
            rationale=(f"Token and numeric evidence overlap: {best_ratio:.0%}." if supported else "No source sufficiently entails this claim."),
        )


class GroqCitationJudge:
    """Strict LLM-as-judge verification through Groq's OpenAI-compatible endpoint."""

    FALLBACK_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b")

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_JUDGE_MODEL")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for Groq citation verification.")
        if not self.model:
            raise ValueError("GROQ_JUDGE_MODEL is required; set it in .env.")

    def judge(self, claim: str, sources: Sequence[SourceChunk]) -> ClaimVerdict:
        from openai import OpenAI

        evidence = "\n\n".join(f"[{source.chunk_id}] {source.text}" for source in sources)
        prompt = f"""You are a strict financial-document citation verifier.
Decide whether every factual or numeric part of CLAIM is directly supported by SOURCE CHUNKS.
Do not use outside knowledge. A related topic is not enough. A claim is supported only if the source entails it.
Return valid JSON only with keys: supported (boolean), source_ids (array of exact source IDs), rationale (one short sentence).

CLAIM:
{claim}

SOURCE CHUNKS:
{evidence}
"""
        client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
        model_candidates = [self.model, *self.FALLBACK_MODELS]
        completion = None
        last_error: Exception | None = None
        for candidate in dict.fromkeys(model_candidates):
            try:
                completion = client.chat.completions.create(
                    model=candidate,
                    temperature=0.01,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                self.model = candidate
                break
            except Exception as exc:  # pragma: no cover - fallback path exercised only on Groq API failures
                message = str(exc)
                if "model_decommissioned" not in message and "no longer supported" not in message:
                    raise
                last_error = exc
        if completion is None:
            raise RuntimeError(f"Groq citation verification failed for all candidate models: {last_error}")
        payload = json.loads(completion.choices[0].message.content or "{}")
        valid_ids = {source.chunk_id for source in sources}
        source_ids = [source_id for source_id in payload.get("source_ids", []) if source_id in valid_ids]
        supported = bool(payload.get("supported")) and bool(source_ids)
        return ClaimVerdict(claim, supported, source_ids if supported else [], str(payload.get("rationale", "No rationale returned.")))


def verify_citations(answer: str, source_chunks: Sequence[SourceChunk], judge: ClaimJudge | None = None) -> VerificationResult:
    """Keep only source-entailed claims and append auditable chunk citations."""
    if not source_chunks:
        raise ValueError("At least one retrieved source chunk is required for verification.")
    verifier = judge or GroqCitationJudge()
    verdicts = [verifier.judge(claim, source_chunks) for claim in extract_claims(answer)]
    retained = [f"{verdict.claim} [source: {', '.join(verdict.source_ids)}]" for verdict in verdicts if verdict.supported]
    verified_answer = " ".join(retained) or "I could not verify the generated claims against the retrieved filing excerpts."
    return VerificationResult(verified_answer, verdicts)
