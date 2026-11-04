"""Grounded calculator tool: the LLM selects operands, Python does the arithmetic.

The model returns operands (each with the exact cell text and the chunk it came from) plus
an arithmetic expression over operand names. Every operand is checked to literally appear
in its cited chunk before anything is computed; on failure the model gets the error back
once and may correct itself. The LLM never performs arithmetic itself.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import asdict, dataclass, field
from typing import Sequence

from src.llm import LLM, LLMError
from src.retrieval.store import Hit

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
NUMBER = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?")


@dataclass
class Operand:
    name: str
    label: str
    value_text: str
    value: float
    chunk_id: str


@dataclass
class Calculation:
    label: str
    expression: str
    result: float
    unit: str
    operands: list[Operand]
    components: list[dict] = field(default_factory=list)  # intermediate values, e.g. each period's margin

    @property
    def formatted(self) -> str:
        if self.unit == "pp":
            return f"{self.result:+.2f} percentage points"
        if self.unit == "%":
            return f"{self.result:.2f}%"
        return f"{self.result:,.2f} {self.unit}".strip()

    def as_evidence(self) -> str:
        parts = "; ".join(f"{o.name} = {o.value_text} ({o.label}, source {o.chunk_id})" for o in self.operands)
        steps = "".join(f" {c['label']}: {c['value']:,.2f}{'%' if c.get('percent') else ''}." for c in self.components)
        return (f"Deterministic calculation by the calculator tool. {self.label}: {self.formatted}.{steps} "
                f"Formula: {self.expression}, where {parts}.")

    def to_dict(self) -> dict:
        return {**asdict(self), "formatted": self.formatted}


def parse_number(text: str) -> float:
    clean = text.strip().replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    negative = clean.startswith("(") and clean.endswith(")")
    clean = clean.strip("()")
    if clean.startswith("-"):
        negative, clean = True, clean[1:]
    value = float(clean)
    return -value if negative else value


def _digits(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def appears_in(value_text: str, chunk_text: str) -> bool:
    """True if the number in value_text occurs as a whole number token in chunk_text."""
    core = _digits(value_text.replace("$", "").replace("%", "").replace("(", "").replace(")", "").strip().lstrip("-"))
    if not re.fullmatch(r"\d+(\.\d+)?", core):
        return False
    return re.search(rf"(?<![\d.]){re.escape(core)}(?![\d]|\.\d)", _digits(chunk_text)) is not None


def safe_eval(expression: str, names: dict[str, float]) -> float:
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")

    return walk(ast.parse(expression, mode="eval"))


CALC_SYSTEM = """You extract operands from SEC filing excerpts for a calculator. You never do arithmetic yourself.
Return JSON: {"operands": [{"name": "a", "label": "<metric, segment, period>", "value_text": "<number exactly as printed>",
"chunk_id": "<id of the excerpt it came from>"}], "expression": "<arithmetic over operand names>",
"unit": "%" | "pp" | "<currency/unit such as USD millions>", "label": "<what the result measures, with periods>",
"components": [{"label": "<intermediate value, e.g. FY2025 operating margin>", "expression": "a / b * 100", "percent": true}]}
List as components the intermediate values a reader needs (e.g. each period's margin when the result is their difference).

Conventions:
- Growth rate: (new - old) / old * 100 with unit "%".
- Change in a margin or percentage: new_pct - old_pct with unit "pp". If a margin is not printed, compute it from its
  components, e.g. (a / b - c / d) * 100 for operating income / revenue in two periods.
- Use the two most recent fiscal years unless the question names others. Prefer annual totals from financial statement
  tables; do not mix segments, quarters, or periods.
- Every operand is a raw figure printed in an excerpt, with value_text copied exactly, e.g. "128,725" or "46.9%".
  Never put a ratio, margin, or growth rate you derived yourself in an operand: express that derivation in the
  expression instead. Use only + - * / and parentheses.
If the excerpts do not contain the needed figures return {"operands": [], "error": "<what is missing>"}."""


class Calculator:
    def __init__(self, llm: LLM | None = None) -> None:
        from src.llm import agent

        self.llm = llm or agent()

    def _validate(self, raw: dict, by_id: dict[str, Hit]) -> Calculation:
        operands_raw = raw.get("operands") or []
        if not operands_raw:
            raise ValueError(raw.get("error") or "no operands returned")
        operands: list[Operand] = []
        for item in operands_raw:
            name, text, cid = str(item.get("name", "")), str(item.get("value_text", "")), str(item.get("chunk_id", ""))
            if not re.fullmatch(r"[a-z]\w*", name):
                raise ValueError(f"invalid operand name {name!r}")
            hit = by_id.get(cid)
            if hit is None:
                raise ValueError(f"operand {name} cites unknown excerpt {cid!r}")
            if not appears_in(text, hit.chunk.text):
                raise ValueError(f"operand {name} value {text!r} does not appear in excerpt {cid}")
            operands.append(Operand(name, str(item.get("label", ""))[:160], text, parse_number(text), cid))
        expression = str(raw.get("expression", ""))
        result = safe_eval(expression, {o.name: o.value for o in operands})
        unit = str(raw.get("unit") or "")
        names = {o.name: o.value for o in operands}
        components = []
        for item in raw.get("components") or []:
            try:  # components are presentation only; a malformed one is skipped rather than failing the calculation
                components.append({"label": str(item["label"])[:120], "value": safe_eval(str(item["expression"]), names),
                                   "percent": bool(item.get("percent"))})
            except (KeyError, ValueError, ZeroDivisionError, SyntaxError, TypeError):
                continue
        return Calculation(str(raw.get("label", ""))[:200], expression, result, unit, operands, components)

    def run(self, question: str, hits: Sequence[Hit]) -> tuple[Calculation | None, list[str]]:
        """Return the calculation (or None) and a log of attempts for the trace."""
        # Highest-scoring tables first, then prose; bounded to stay inside the per-minute token budget.
        tables_first = sorted(hits, key=lambda h: (h.chunk.kind != "table", -(h.rerank_score or 0)))[:6]
        by_id = {h.chunk.chunk_id: h for h in tables_first}
        excerpts = "\n\n".join(f"[{h.chunk.chunk_id}] {h.chunk.context}\n{h.chunk.text[:2000]}" for h in tables_first)
        latest = max((h.chunk.fiscal_year for h in tables_first), default=None)
        period_hint = f"The latest fiscal year in these filings is {latest}; 'latest' or 'year over year' means {latest} vs {latest - 1}.\n" if latest else ""
        user = f"Question: {question}\n{period_hint}\nExcerpts:\n{excerpts}"
        log: list[str] = []
        for attempt in range(3):
            try:
                raw = self.llm.complete_json(CALC_SYSTEM, user)
                calc = self._validate(raw, by_id)
                log.append(f"attempt {attempt + 1}: ok -> {calc.formatted}")
                return calc, log
            except (LLMError, ValueError, ZeroDivisionError, SyntaxError) as exc:
                log.append(f"attempt {attempt + 1}: rejected ({exc})")
                # Self-correction: show the model exactly why its operands were rejected.
                user = (f"{user}\n\nYour previous answer was rejected: {exc}. Operands must be numbers printed in the "
                        "excerpts (copied exactly); put any ratio or percentage you need into the expression.")
        return None, log
