from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..construction import ConstructionExperiment, candidate_from_dict, decision_to_dict


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset" / "memory_construction.json"
OUTPUT = ROOT / "experiments" / "results" / "construction" / "latest"


def run(dataset: Path = DATASET, output: Path = OUTPUT) -> dict[str, Any]:
    """Run all construction strategies and write portable result artifacts."""
    raw_candidates = json.loads(dataset.read_text(encoding="utf-8"))
    candidates = [candidate_from_dict(item) for item in raw_candidates]
    result = ConstructionExperiment(candidates).run()
    output.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": result["dataset"],
        "strategies": {
            name: {"metrics": strategy_result["metrics"]}
            for name, strategy_result in result["strategies"].items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output / "decisions.jsonl").open("w", encoding="utf-8") as fh:
        for strategy_result in result["strategies"].values():
            for decision in strategy_result["decisions"]:
                serialized = json.dumps(decision_to_dict(decision), ensure_ascii=False)
                fh.write(serialized + "\n")

    fields = [
        "strategy", "precision", "recall", "f1", "routing_accuracy",
        "routing_coverage", "saved_count", "total_stored_tokens",
        "token_savings_vs_store_all",
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name, strategy_result in result["strategies"].items():
            metrics = strategy_result["metrics"]
            row = {
                "strategy": name,
                **{key: metrics.get(key) for key in fields if key != "strategy"},
            }
            writer.writerow(row)

    lines = [
        "# Memory Construction Evaluation",
        "",
        f"候选数：{result['dataset']['candidate_count']}",
        "",
        "| strategy | precision | recall | F1 | saved | tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, strategy_result in result["strategies"].items():
        metrics = strategy_result["metrics"]
        lines.append(
            f"| {name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} "
            f"| {metrics['f1']:.3f} | {metrics['saved_count']} "
            f"| {metrics['total_stored_tokens']} |"
        )
    lines.extend([
        "",
        "数据集是可复现的开发/演示标注集，不代表独立人工 benchmark。",
        "Token 为确定性估算，不是供应商 tokenizer 的真实计费值。",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    for name, strategy_result in result["strategies"].items():
        print(name, json.dumps(strategy_result["metrics"], ensure_ascii=False))
