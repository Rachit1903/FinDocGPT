"""Evaluation metrics. Deterministic where possible; an independent LLM judge where not.

Deterministic:  route accuracy, evidence recall (gold evidence strings found in the context the
                generator saw), numeric accuracy (computation answers vs. hand-computed values),
                abstention on unanswerable questions.
LLM judge:      key-fact coverage against the gold reference, and faithfulness (share of answer
                sentences supported by the system's own retrieved context). The judge model is a
                different family from the generator and never sees which system produced the answer.
"""

from __future__ import annotations

import re
from typing import Any

from src.agent.answer import is_abstention, split_sentences
from src.llm import LLM, LLMError

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‑": "-", "–": "-", "—": "-"})


def normalize(text: str) -> str:
    return " ".join(text.translate(QUOTES).lower().replace("$", "").split())


def evidence_found(evidence: str, context: str) -> bool:
    """Substring match for prose; for a table row ("| label | v1 | v2 |") the label and every number must be
    present, so systems that render tables differently (e.g. v1's raw dataframe text) are scored fairly."""
    ctx = normalize(context)
    if not evidence.startswith("|"):
        return normalize(evidence) in ctx
    cells = [c.strip() for c in evidence.strip("|").split("|")]
    numbers = [n.replace(",", "") for c in cells[1:] for n in re.findall(r"\d[\d,]*\.?\d*", c)]
    digits = ctx.replace(",", "")
    return normalize(cells[0]) in ctx and all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", digits) for n in numbers)


def evidence_recall(evidence: list[str], contexts: list[str]) -> float | None:
    if not evidence:
        return None
    return sum(any(evidence_found(e, c) for c in contexts) for e in evidence) / len(evidence)


def numeric_correct(answer: str, expected: dict[str, Any]) -> bool:
    target = abs(float(expected["value"]))
    tolerance = max(0.15, 0.02 * target)
    for token in NUMBER.findall(answer.replace("−", "-")):
        try:
            value = abs(float(token.replace(",", "").rstrip(".")))
        except ValueError:
            continue
        if abs(value - target) <= tolerance:
            return True
    return False


JUDGE_SYSTEM = """You grade answers to questions about SEC 10-K filings. Be strict and literal.
Given a QUESTION, a GOLD reference answer with numbered KEY FACTS, a candidate ANSWER split into numbered sentences, and the
CONTEXT the candidate system retrieved, return JSON:
{"key_facts": [{"id": <n>, "covered": true|false}],
 "contradicts_gold": true|false,
 "sentences": [{"id": <n>, "supported_by_context": true|false}]}
- A key fact is covered if the answer states it (paraphrase and reasonable rounding are fine; a wrong number is not covered).
- contradicts_gold is true if any answer statement conflicts with the gold answer.
- supported_by_context: every fact/number in the sentence appears in or follows directly from the CONTEXT (not your own knowledge).
  Sentences that only say information is unavailable count as supported."""


def judge(llm: LLM, question: str, gold: dict[str, Any], answer: str, contexts: list[str]) -> dict[str, Any]:
    sentences = split_sentences(answer) or [answer]
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(gold["key_facts"], 1))
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    context = "\n\n".join(c[:1500] for c in contexts)[:9000]
    user = (f"QUESTION: {question}\n\nGOLD: {gold['reference_answer']}\nKEY FACTS:\n{facts}\n\n"
            f"ANSWER SENTENCES:\n{numbered}\n\nCONTEXT:\n{context}")
    try:
        raw = llm.complete_json(JUDGE_SYSTEM, user)
    except LLMError as exc:
        return {"error": str(exc)}
    covered = [bool(f.get("covered")) for f in raw.get("key_facts", [])][: len(gold["key_facts"])]
    supported = [bool(s.get("supported_by_context")) for s in raw.get("sentences", [])][: len(sentences)]
    return {
        "fact_coverage": sum(covered) / len(gold["key_facts"]) if gold["key_facts"] else None,
        "contradicts_gold": bool(raw.get("contradicts_gold")),
        "faithfulness": sum(supported) / len(supported) if supported else None,
    }


def score_item(gold: dict[str, Any], pred: dict[str, Any], judge_llm: LLM | None) -> dict[str, Any]:
    answer = pred.get("answer", "") or ""
    abstained = is_abstention(answer) or answer.startswith("I could not verify")
    row: dict[str, Any] = {"id": gold["id"], "category": gold["category"], "abstained": abstained}
    if gold["category"] == "unanswerable":
        row["abstention_correct"] = abstained
        return row
    row["route_correct"] = pred.get("route") == gold["category"]
    row["evidence_recall"] = evidence_recall(gold["evidence"], pred.get("contexts", []))
    if gold.get("expected_value"):
        row["numeric_correct"] = numeric_correct(answer, gold["expected_value"])
    if judge_llm is not None:
        verdict = judge(judge_llm, gold["question"], gold, answer, pred.get("contexts", []))
        row.update(verdict)
        if "fact_coverage" in verdict:
            row["correct"] = verdict["fact_coverage"] >= 0.67 and not verdict["contradicts_gold"]
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str, subset: list[dict[str, Any]]) -> float | None:
        values = [r[key] for r in subset if r.get(key) is not None]
        return round(sum(map(float, values)) / len(values), 3) if values else None

    answerable = [r for r in rows if r["category"] != "unanswerable"]
    summary: dict[str, Any] = {"overall": {
        "n": len(answerable),
        "correct": mean("correct", answerable),
        "fact_coverage": mean("fact_coverage", answerable),
        "faithfulness": mean("faithfulness", answerable),
        "evidence_recall": mean("evidence_recall", answerable),
        "route_accuracy": mean("route_correct", answerable),
        "numeric_accuracy": mean("numeric_correct", answerable),
        "false_abstention": mean("abstained", answerable),
        "abstention_on_unanswerable": mean("abstention_correct", [r for r in rows if r["category"] == "unanswerable"]),
    }}
    for category in ("factual", "comparison", "computation"):
        subset = [r for r in rows if r["category"] == category]
        summary[category] = {k: mean(k, subset) for k in ("correct", "fact_coverage", "faithfulness", "evidence_recall", "route_correct", "numeric_correct")}
        summary[category]["n"] = len(subset)
    return summary
