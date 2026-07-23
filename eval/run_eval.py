"""Run RAGAS with Groq 70B as judge and local BGE embeddings; never OpenAI."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "eval" / "test_set.json"
RESULTS_DIR = ROOT / "eval" / "results"
SUMMARY_COLUMNS = ["faithfulness", "llm_context_precision_with_reference", "answer_relevancy"]
FALLBACK_JUDGE_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b")


def load_test_set() -> list[dict]:
    if not TEST_SET.exists():
        raise FileNotFoundError(f"Create {TEST_SET.name} from the reviewed draft first.")
    items = json.loads(TEST_SET.read_text(encoding="utf-8"))
    if len(items) < 40:
        raise ValueError(f"Evaluation requires 40 reviewed items; found {len(items)}.")
    missing = [item.get("id", "unknown") for item in items if not item.get("reference_answer") or not item.get("reference_source_ids")]
    if missing:
        raise ValueError(f"Items need verified reference_answer and reference_source_ids before scoring: {', '.join(missing[:5])}")
    return items


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in SUMMARY_COLUMNS if column in frame.columns]


def _format_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)

    def format_cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _summarize_results(frame: pd.DataFrame, categories: list[str]) -> tuple[pd.DataFrame, str]:
    metric_columns = _metric_columns(frame)
    if not metric_columns:
        raise ValueError("RAGAS output did not contain the expected metric columns.")

    annotated = frame.copy()
    annotated.insert(0, "category", categories)

    summary_rows: list[dict[str, object]] = []
    summary_rows.append({"scope": "overall", "samples": len(annotated), **{column: annotated[column].mean() for column in metric_columns}})
    for category, group in annotated.groupby("category", sort=True):
        summary_rows.append({"scope": category, "samples": len(group), **{column: group[column].mean() for column in metric_columns}})

    summary = pd.DataFrame(summary_rows)
    best_metric = max(metric_columns, key=lambda column: summary.loc[summary["scope"] == "overall", column].iloc[0])
    worst_metric = min(metric_columns, key=lambda column: summary.loc[summary["scope"] == "overall", column].iloc[0])
    overall = summary.loc[summary["scope"] == "overall"].iloc[0]
    note = (
        f"Overall strongest metric is {best_metric} at {overall[best_metric]:.3f}. "
        f"The weakest overall metric is {worst_metric} at {overall[worst_metric]:.3f}. "
        "Category rows below show where factual, comparison, and computation queries diverge."
    )
    return summary, note


def _resolve_judge_model() -> str:
    configured = os.getenv("GROQ_JUDGE_MODEL")
    if not configured:
        raise ValueError("GROQ_JUDGE_MODEL is required for RAGAS.")
    if configured in {"llama-3.1-70b-versatile", "llama-3.1-70b-specdec"}:
        return FALLBACK_JUDGE_MODELS[0]
    return configured


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY") or not os.getenv("GROQ_JUDGE_MODEL"):
        raise ValueError("GROQ_API_KEY and GROQ_JUDGE_MODEL are required for RAGAS.")
    items = load_test_set()

    # Imports are kept here so draft validation works before the expensive eval dependencies are installed.
    from openai import OpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import HuggingfaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference
    from src.generation.groq import GroqGenerator
    from src.router.pipeline import FinDocPipeline
    from src.router.query_router import QueryRoute

    pipeline = FinDocPipeline()
    generator = GroqGenerator()
    samples = []
    categories: list[str] = []
    for item in items:
        pipeline_result = pipeline.answer(item["question"])
        categories.append(item["category"])
        if pipeline_result.route is QueryRoute.COMPUTATION:
            numeric_answer = pipeline_result.result
            contexts = [numeric_answer.source_context]
            response = numeric_answer.answer
        else:
            ranked = pipeline_result.result
            contexts = [result.chunk.text for result in ranked]
            response = generator.answer(item["question"], contexts)
        samples.append(SingleTurnSample(user_input=item["question"], response=response, reference=item["reference_answer"], retrieved_contexts=contexts))

    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    judge = llm_factory(_resolve_judge_model(), provider="openai", client=client)
    embeddings = HuggingfaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    result = evaluate(EvaluationDataset(samples=samples), metrics=[Faithfulness(llm=judge), LLMContextPrecisionWithReference(llm=judge), AnswerRelevancy(llm=judge, embeddings=embeddings)])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result_frame = result.to_pandas()
    result_path = RESULTS_DIR / f"ragas_{stamp}.csv"
    result_frame.to_csv(result_path, index=False)
    summary_frame, note = _summarize_results(result_frame, categories)
    summary_path = RESULTS_DIR / f"ragas_{stamp}_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    summary_md = RESULTS_DIR / f"ragas_{stamp}_summary.md"
    summary_md.write_text(_format_markdown_table(summary_frame), encoding="utf-8")
    notes_path = RESULTS_DIR / f"ragas_{stamp}_notes.md"
    notes_path.write_text(note + "\n", encoding="utf-8")
    print(f"Saved raw results to {result_path}")
    print(f"Saved summary table to {summary_path}")
    print(f"Saved summary note to {notes_path}")
    print(note)


if __name__ == "__main__":
    main()
