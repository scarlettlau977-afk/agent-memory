from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .framework import AgentMemory
from .models import MemoryTier


@dataclass
class ConstructionCase:
    content: str
    should_remember: bool
    tier: MemoryTier | None = None
    user_id: str = "eval-user"
    session_id: str = "eval-session"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalCase:
    query: str
    relevant_memory_ids: set[str]
    user_id: str = "eval-user"
    session_id: str | None = "eval-session"
    top_k: int = 5


@dataclass
class EvaluationReport:
    metrics: dict[str, float]
    details: list[dict[str, Any]] = field(default_factory=list)


def evaluate_construction(policy_or_memory: Any, cases: Iterable[ConstructionCase]) -> EvaluationReport:
    policy = getattr(policy_or_memory, "policy", policy_or_memory)
    details, tp, fp, fn, tier_ok, tier_total = [], 0, 0, 0, 0, 0
    for case in cases:
        decision = policy.decide(case.content, user_id=case.user_id, session_id=case.session_id, metadata=case.metadata)
        if decision.should_remember and case.should_remember: tp += 1
        elif decision.should_remember: fp += 1
        elif case.should_remember: fn += 1
        if case.should_remember and decision.should_remember and case.tier is not None:
            tier_total += 1; tier_ok += int(decision.tier == case.tier)
        details.append({"content": case.content, "expected": case.should_remember, "predicted": decision.should_remember,
                        "predicted_tier": decision.tier.value if decision.tier else None})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return EvaluationReport({"precision": precision, "recall": recall,
                             "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                             "tier_accuracy": tier_ok / tier_total if tier_total else 0.0}, details)


def evaluate_retrieval(memory: AgentMemory, cases: Iterable[RetrievalCase]) -> EvaluationReport:
    details, recalls, mrrs, ndcgs = [], [], [], []
    for case in cases:
        results = memory.retrieve(case.query, user_id=case.user_id, session_id=case.session_id, top_k=case.top_k)
        ids = [r.memory.id for r in results]
        hits = [i for i, item_id in enumerate(ids) if item_id in case.relevant_memory_ids]
        recalls.append(float(bool(hits)))
        mrrs.append(1 / (hits[0] + 1) if hits else 0.0)
        dcg = sum(1 / __import__("math").log2(i + 2) for i in hits)
        ideal = sum(1 / __import__("math").log2(i + 2) for i in range(min(len(case.relevant_memory_ids), case.top_k)))
        ndcgs.append(dcg / ideal if ideal else 0.0)
        details.append({"query": case.query, "returned_ids": ids, "hits": len(hits)})
    n = max(len(details), 1)
    return EvaluationReport({"recall@k": sum(recalls) / n, "mrr": sum(mrrs) / n, "ndcg@k": sum(ndcgs) / n}, details)


def evaluate_downstream(cases: Iterable[Any], run_agent: Callable[[Any, str], Any],
                        score: Callable[[Any, Any], float]) -> EvaluationReport:
    """Compare an agent with memory against a no-memory/control context."""
    deltas, details = [], []
    for case in cases:
        with_memory = run_agent(case, getattr(case, "memory_context", ""))
        without_memory = run_agent(case, "")
        delta = float(score(case, with_memory)) - float(score(case, without_memory))
        deltas.append(delta)
        details.append({"case": case, "with_memory": with_memory, "without_memory": without_memory, "delta": delta})
    return EvaluationReport({"mean_score_gain": sum(deltas) / max(len(deltas), 1), "n": float(len(deltas))}, details)
