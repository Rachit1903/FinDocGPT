# FinDocGPT evaluation report

Gold set: 45 questions (40 answerable: 20 factual, 10 comparison, 10 computation; 5 unanswerable).

| Metric | v1 | v2 |
|---|---|---|
| Answer correctness (judge: ≥2/3 key facts, no contradiction) | 0.17 | 0.55 |
| Key-fact coverage | 0.28 | 0.75 |
| Faithfulness to retrieved context | 0.78 | 0.97 |
| Numeric accuracy (computation, exact value) | 0.20 | 1.00 |
| Gold evidence recall in context | 0.15 | 0.80 |
| Route accuracy | 0.65 | 1.00 |
| False abstention (lower is better) | 0.15 | 0.03 |
| Abstains on unanswerable questions | 0.60 | 1.00 |

## By category (answer correctness / evidence recall)

| Category | v1 | v2 |
|---|---|---|
| factual | 0.05 / 0.10 | 0.35 / 0.82 |
| comparison | 0.40 / 0.20 | 0.60 / 0.65 |
| computation | 0.20 / 0.20 | 0.90 / 0.90 |

Judge model: `qwen/qwen3.8-27b`; generator: `openai/gpt-oss-120b`.
