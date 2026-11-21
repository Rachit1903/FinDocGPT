"""Cited answer generation and claim-level verification."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.llm import LLM, LLMError

ABSTAIN_PATTERN = re.compile(r"(do(es)? not (contain|provide|disclose|include)|not (disclosed|provided|available|included) in"
                             r"|insufficient|cannot (be )?(answer|determine)|no information)", re.I)
CITATION = re.compile(r"\[S(\d+)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z$(])|\n+(?=\s*[-*•]?\s*\S)")

GENERATE_SYSTEM = """You are a financial analyst answering questions strictly from SEC 10-K excerpts.
Rules:
- Use only the numbered sources. Every sentence that states a fact or number must end with citations like [S2] or [S1][S3].
- If a source is a deterministic calculation, report its result exactly as given and cite it, together with the
  underlying reported figures (do not mention the calculator itself).
- Never do arithmetic yourself: only state figures that appear in the sources.
- Be concise: 2-5 sentences of plain prose, no headings, no tables, no bullet lists. Include the key figures and periods.
- Name fiscal years as the filing does (e.g. "fiscal 2025").
- If the sources do not contain the answer, reply exactly: "The filings in the corpus do not disclose this." and nothing else."""

VERIFY_SYSTEM = """You verify an answer sentence by sentence against SEC filing sources.
A sentence is supported only if every fact and number in it is stated in, or directly follows from, the sources
(rounding such as 19.7% for 19.68% is fine; a deterministic calculation source counts as evidence).
Return JSON: {"verdicts": [{"id": <sentence id>, "supported": true|false, "reason": "<short>"}]} covering every sentence."""


@dataclass
class Verdict:
    sentence: str
    supported: bool
    citations: list[int]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


TYPOGRAPHY = str.maketrans({" ": " ", " ": " ", "‑": "-", " ": " "})


def normalize_citations(text: str) -> str:
    """gpt-oss sometimes emits its native citation style (【S4†L19-L23】) and narrow no-break spaces; normalize both."""
    return re.sub(r"\s*【\s*S(\d+)[^】]*】", r"[S\1]", text.translate(TYPOGRAPHY))


def split_sentences(text: str) -> list[str]:
    return [s.strip(" -*•") for s in SENTENCE_SPLIT.split(text.strip()) if s.strip(" -*•")]


AMOUNT = r"(\$\s?\(?[\d,]+(?:\.\d+)?\)?(?:\s*(?:billion|million|thousand))?|\(?[\d,]+(?:\.\d+)?\)?\s?%)"
UP_WORDS = r"(?:increas\w*|rose|rise|grew|grow\w*|up|higher|improv\w*|expand\w*)"
DOWN_WORDS = r"(?:decreas\w*|fell|fall\w*|declin\w*|down|lower|dropp?\w*|contract\w*)"
# "<word> from A to B" and "B (<word> from A)" — the two ways answers state a change between reported figures.
FROM_TO = re.compile(rf"\b({UP_WORDS}|{DOWN_WORDS})\b[^.;]{{0,30}}?\bfrom\s+{AMOUNT}[^.;]{{0,50}}?\bto\s+{AMOUNT}", re.I)
B_FROM_A = re.compile(rf"{AMOUNT}[^.;]{{0,50}}?\(\s*({UP_WORDS}|{DOWN_WORDS})\s+from\s+{AMOUNT}", re.I)
SCALE = {"thousand": 1e-3, "million": 1.0, "billion": 1e3}


def _amount(text: str) -> tuple[float, str] | None:
    """Parse '$7.2 billion' / '(6)%' / '$(2,656)' into (value in millions or percent, kind)."""
    kind = "%" if "%" in text else "$"
    scale = next((v for k, v in SCALE.items() if k in text.lower()), 1.0)
    digits = re.search(r"[\d,]+(?:\.\d+)?", text)
    if not digits:
        return None
    value = float(digits.group().replace(",", "")) * (scale if kind == "$" else 1.0)
    return (-value if "(" in text else value), kind


def direction_conflict(sentence: str) -> str | None:
    """Deterministic check: a direction word that contradicts the two figures it relates ("$7.2B, down from $3.7B")."""
    checks = [(m.group(1), m.group(2), m.group(3)) for m in FROM_TO.finditer(sentence)]
    checks += [(m.group(2), m.group(3), m.group(1)) for m in B_FROM_A.finditer(sentence)]
    for word, old_text, new_text in checks:
        old, new = _amount(old_text), _amount(new_text)
        if not old or not new or old[1] != new[1] or old[0] == new[0]:
            continue
        says_up = re.fullmatch(UP_WORDS, word, re.I) is not None
        if says_up != (new[0] > old[0]):
            return f"'{word}' contradicts the figures ({old_text.strip()} -> {new_text.strip()})"
    return None


def is_abstention(text: str) -> bool:
    return bool(ABSTAIN_PATTERN.search(text)) and len(text) < 400


class AnswerWriter:
    def __init__(self, llm: LLM | None = None) -> None:
        from src.llm import generator

        self.llm = llm or generator()

    def write(self, question: str, sources: list[tuple[str, str]], feedback: list["Verdict"] | None = None) -> str:
        block = "\n\n".join(f"[S{i}] {header}\n{text}" for i, (header, text) in enumerate(sources, 1))
        user = f"Question: {question}\n\nSources:\n{block}"
        if feedback:
            # Revision pass: the verifier's objections are returned so the writer can fix or drop those statements.
            issues = "\n".join(f"- {v.sentence} -> {v.reason}" for v in feedback)
            user += (f"\n\nA fact-checker rejected these sentences of your previous answer:\n{issues}\n"
                     "Write a corrected answer: keep only statements the sources support, with correct citations.")
        return normalize_citations(self.llm.complete(GENERATE_SYSTEM, user).strip())


class Verifier:
    def __init__(self, llm: LLM | None = None) -> None:
        from src.llm import agent

        self.llm = llm or agent()

    def verify(self, answer: str, sources: list[tuple[str, str]]) -> list[Verdict]:
        sentences = split_sentences(answer)
        if not sentences or (len(sentences) == 1 and is_abstention(sentences[0])):
            return [Verdict(s, True, [], "abstention") for s in sentences]
        # All sources are shown: a correct fact with a wrong citation should be fixed, not deleted.
        block = "\n\n".join(f"[S{i}] {header}\n{text}" for i, (header, text) in enumerate(sources, 1))
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
        try:
            raw = self.llm.complete_json(VERIFY_SYSTEM, f"Sources:\n{block}\n\nAnswer sentences:\n{numbered}")
            by_id = {int(v.get("id")): v for v in raw.get("verdicts", []) if str(v.get("id", "")).isdigit()}
        except (LLMError, ValueError, TypeError):
            by_id = {}
        verdicts = []
        for i, sentence in enumerate(sentences, 1):
            cites = [int(n) for n in CITATION.findall(sentence)]
            item = by_id.get(i)
            conflict = direction_conflict(sentence)
            if conflict:
                # Overrides the LLM verdict: the verifier model missed exactly this in the eval (NVIDIA provisions).
                verdicts.append(Verdict(sentence, False, cites, conflict))
            elif item is None:
                # Unverifiable (verifier failed): keep only sentences that cite a source.
                verdicts.append(Verdict(sentence, bool(cites), cites, "not verified (verifier unavailable)"))
            else:
                verdicts.append(Verdict(sentence, bool(item.get("supported")), cites, str(item.get("reason", ""))[:200]))
        return verdicts
