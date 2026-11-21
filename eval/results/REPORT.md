# FinDocGPT evaluation report

Gold set: 45 questions (40 answerable: 20 factual, 10 comparison, 10 computation; 5 unanswerable).

| Metric | v1 | v2-run1 | v2 | v2-tablenotes |
|---|---|---|---|---|
| Answer correctness (judge: ≥2/3 key facts, no contradiction) | 0.17 | 0.47 | 0.55 | 0.50 |
| Key-fact coverage | 0.28 | 0.66 | 0.75 | 0.72 |
| Faithfulness to retrieved context | 0.78 | 0.95 | 0.97 | 0.94 |
| Numeric accuracy (computation, exact value) | 0.20 | 1.00 | 1.00 | 0.90 |
| Gold evidence recall in context | 0.15 | 0.76 | 0.80 | 0.76 |
| Route accuracy | 0.65 | 1.00 | 1.00 | 1.00 |
| False abstention (lower is better) | 0.15 | 0.05 | 0.03 | 0.05 |
| Abstains on unanswerable questions | 0.60 | 1.00 | 1.00 | 1.00 |

## By category (answer correctness / evidence recall)

| Category | v1 | v2-run1 | v2 | v2-tablenotes |
|---|---|---|---|---|
| factual | 0.05 / 0.10 | 0.40 / 0.78 | 0.35 / 0.82 | 0.40 / 0.72 |
| comparison | 0.40 / 0.20 | 0.50 / 0.60 | 0.60 / 0.65 | 0.40 / 0.60 |
| computation | 0.20 / 0.20 | 0.60 / 0.90 | 0.90 / 0.90 | 0.80 / 1.00 |

Judge model: `qwen/qwen3.8-27b`; generator: `openai/gpt-oss-120b`.
- **v2-run1**: Agent before the final changes (verifier saw cited sources only; no revision loop; calculator without intermediate components/period hint; planner without MD&A hint).
- **v2-tablenotes**: Experiment, reverted: table chunks carried the preceding prose as a "Note:" line, plus stricter writer/verifier prompts about describing figures. Correctness fell 0.55 -> 0.50: notes attached unrelated MD&A text to tables without an introduction (the calculator then swapped Amazon's year columns), the stricter verifier rejected a partly correct answer, and two answers rounded the calculator result.
