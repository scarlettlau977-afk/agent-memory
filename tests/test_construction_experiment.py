import json
from pathlib import Path

from agent_memory.experiments.memory_construction import run


def test_experiment_writes_reproducible_outputs(tmp_path):
    dataset = Path(__file__).parents[1] / "dataset" / "memory_construction.json"
    result = run(dataset, tmp_path)
    assert set(result["strategies"]) == {"store_all", "heuristic", "explicit_only", "quality_aware"}
    for name in ("summary.json", "decisions.jsonl", "comparison.csv", "report.md"):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset"]["candidate_count"] == 30
