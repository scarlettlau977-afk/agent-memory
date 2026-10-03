from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from ..construction import ConstructionExperiment, candidate_from_dict, decision_to_dict


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset" / "memory_construction.json"
OUTPUT = ROOT / "experiments" / "results" / "construction" / "latest"


def run(dataset: Path = DATASET, output: Path = OUTPUT) -> dict:
    candidates = [candidate_from_dict(item) for item in json.loads(dataset.read_text(encoding="utf-8"))]
    result = ConstructionExperiment(candidates).run()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"dataset": result["dataset"], "strategies": {k: {"metrics": v["metrics"]} for k, v in result["strategies"].items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as fh:
        for name, value in result["strategies"].items():
            for decision in value["decisions"]:
                fh.write(json.dumps(decision_to_dict(decision), ensure_ascii=False) + "\n")
    fields = ["strategy", "precision", "recall", "f1", "routing_accuracy", "routing_coverage", "saved_count", "total_stored_tokens", "token_savings_vs_store_all"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name, value in result["strategies"].items():
            row = {"strategy": name, **{key: value["metrics"].get(key) for key in fields if key != "strategy"}}
            writer.writerow(row)
    lines = ["# Memory Construction Evaluation", "", f"候选数：{result['dataset']['candidate_count']}", "", "| strategy | precision | recall | F1 | saved | tokens |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in result["strategies"].items():
        m = value["metrics"]
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['saved_count']} | {m['total_stored_tokens']} |")
    lines += ["", "数据集是可复现的开发/演示标注集，不代表独立人工 benchmark。Token 为确定性估算，不是供应商 tokenizer 的真实计费值。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    for name, value in result["strategies"].items():
        print(name, json.dumps(value["metrics"], ensure_ascii=False))
