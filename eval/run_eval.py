"""Run and score FinDocGPT against the hand-verified gold set.

    python -m eval.run_eval                      # run the v2 agent, judge, write results + report
    python -m eval.run_eval --system v1          # re-score the frozen v1 baseline predictions
    python -m eval.run_eval --config no_verify   # ablation (see CONFIGS)
    python -m eval.run_eval --report             # rebuild eval/results/REPORT.md from saved scores

Predictions are cached per item, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import settings
from src.llm import USAGE, judge

from .metrics import score_item, summarize

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
GOLD = EVAL_DIR / "gold_set.json"
V1_ROUTES = {"factual_lookup": "factual", "comparison": "comparison", "computation": "computation"}
CONFIGS = {
    "v2": {},
    "no_rewrite": {"enable_rewrite": False},
    "no_verify": {"enable_verify": False},
    "rules_planner": {"rules_planner": True},
}


def load_gold() -> list[dict]:
    return json.loads(GOLD.read_text(encoding="utf-8"))


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def build_agent(config: str):
    from src.agent.graph import FinDocAgent
    from src.agent.planner import Planner, rule_based_plan

    options = dict(CONFIGS[config])
    planner = None
    if options.pop("rules_planner", False):
        planner = Planner()
        planner.plan = rule_based_plan  # type: ignore[method-assign]
    return FinDocAgent(planner=planner, **options)


def run_system(config: str, gold: list[dict]) -> list[dict]:
    path = RESULTS / f"{config}_predictions.json"
    done = {p["id"]: p for p in _read(path)}
    agent = None
    for item in gold:
        if item["id"] in done:
            continue
        agent = agent or build_agent(config)
        started = time.perf_counter()
        result = agent.run(item["question"])
        done[item["id"]] = {
            "id": item["id"], "question": item["question"], "route": result["route"], "answer": result["answer"],
            "draft": result["draft"], "contexts": [s["text"] for s in result["sources"]],
            "source_ids": [s["chunk_id"] for s in result["sources"]], "plan": result["plan"],
            "calculation": result["calculation"], "removed_claims": result["removed_claims"], "trace": result["trace"],
            "latency_s": round(time.perf_counter() - started, 1),
        }
        _write(path, list(done.values()))
        print(f"{item['id']} [{result['route']}] {result['answer'][:110]!r}", flush=True)
    return [done[item["id"]] for item in gold if item["id"] in done]


def score(config: str, predictions: list[dict], gold: list[dict], use_judge: bool) -> dict:
    by_id = {p["id"]: p for p in predictions}
    judge_llm = judge() if use_judge else None
    rows = []
    for item in gold:
        pred = dict(by_id[item["id"]])
        pred["route"] = V1_ROUTES.get(pred.get("route"), pred.get("route"))
        rows.append(score_item(item, pred, judge_llm))
    result = {"config": config, "judge_model": settings.judge_model if use_judge else None,
              "generator_model": settings.generator_model, "summary": summarize(rows), "items": rows}
    _write(RESULTS / f"{config}_scores.json", result)
    return result


def write_report() -> None:
    names = ("v1", "v2", "no_rewrite", "no_verify", "rules_planner")
    systems = [(name, _read(RESULTS / f"{name}_scores.json")) for name in names]
    # A partial run (e.g. --ids) is not comparable with full runs, so only complete score files are reported.
    systems = [(n, s) for n, s in systems if s and len(s["items"]) == len(load_gold())]
    metrics = [("correct", "Answer correctness (judge: ≥2/3 key facts, no contradiction)"),
               ("fact_coverage", "Key-fact coverage"), ("faithfulness", "Faithfulness to retrieved context"),
               ("numeric_accuracy", "Numeric accuracy (computation, exact value)"), ("evidence_recall", "Gold evidence recall in context"),
               ("route_accuracy", "Route accuracy"), ("false_abstention", "False abstention (lower is better)"),
               ("abstention_on_unanswerable", "Abstains on unanswerable questions")]
    fmt = lambda v: "—" if v is None else f"{v:.2f}"
    lines = ["# FinDocGPT evaluation report", "",
             f"Gold set: {len(load_gold())} questions (40 answerable: 20 factual, 10 comparison, 10 computation; 5 unanswerable).",
             "", "| Metric | " + " | ".join(n for n, _ in systems) + " |", "|---|" + "---|" * len(systems)]
    for key, label in metrics:
        lines.append(f"| {label} | " + " | ".join(fmt(s["summary"]["overall"].get(key)) for _, s in systems) + " |")
    lines += ["", "## By category (answer correctness / evidence recall)", "",
              "| Category | " + " | ".join(n for n, _ in systems) + " |", "|---|" + "---|" * len(systems)]
    for category in ("factual", "comparison", "computation"):
        cells = [f"{fmt(s['summary'][category]['correct'])} / {fmt(s['summary'][category]['evidence_recall'])}" for _, s in systems]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    judge_model = next((s["judge_model"] for _, s in systems if s.get("judge_model")), None)
    lines += ["", f"Judge model: `{judge_model}`; generator: `{systems[-1][1]['generator_model']}`."]
    lines += [f"- **{n}**: {s['note']}" for n, s in systems if s.get("note")]
    lines.append("")
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=["v1", "v2"], default="v2")
    parser.add_argument("--config", choices=list(CONFIGS), default=None, help="v2 configuration / ablation")
    parser.add_argument("--ids", nargs="*", help="only these gold ids (for quick iteration)")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        write_report()
        return
    gold = load_gold()
    if args.ids:
        gold = [g for g in gold if g["id"] in set(args.ids)]
    if args.system == "v1":
        config, predictions = "v1", [p for p in _read(RESULTS / "v1_predictions.json") if p["id"] in {g["id"] for g in gold}]
    else:
        config = args.config or "v2"
        predictions = run_system(config, gold)
    result = score(config, predictions, gold, use_judge=not args.no_judge)
    print(json.dumps(result["summary"], indent=1))
    print(f"LLM usage: {USAGE}")
    if not args.ids:
        write_report()


if __name__ == "__main__":
    main()
