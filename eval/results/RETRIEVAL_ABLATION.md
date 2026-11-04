# Retrieval ablation

40 answerable gold questions; recall = share of gold evidence strings present in the top-k chunks. Embeddings `BAAI/bge-base-en-v1.5`, reranker `BAAI/bge-reranker-base`.

| Configuration | recall@5 | recall@8 | hit@5 | hit@8 | MRR |
|---|---|---|---|---|---|
| v1 pipeline (frozen contexts) | 0.15 | 0.15 | 0.15 | 0.15 | 0.10 |
| BM25 only | 0.36 | 0.44 | 0.38 | 0.45 | 0.20 |
| Dense only | 0.44 | 0.59 | 0.47 | 0.62 | 0.30 |
| Hybrid (RRF) | 0.40 | 0.55 | 0.42 | 0.57 | 0.25 |
| Hybrid + rerank | 0.62 | 0.66 | 0.65 | 0.68 | 0.41 |
| Hybrid + rerank + company filter | 0.57 | 0.66 | 0.60 | 0.68 | 0.40 |
