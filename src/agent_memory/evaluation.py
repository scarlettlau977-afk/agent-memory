from __future__ import annotations

import math
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
    metrics: dict[str, Any]
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


def _stable_id(value: Any) -> str:
    """Normalize IDs so equivalent numeric and string IDs match consistently."""
    return str(value)


def _result_id(result: Any) -> str:
    """Extract the existing stable memory ID from supported result shapes."""
    if isinstance(result, dict):
        if "memory" in result:
            return _result_id(result["memory"])
        if "id" in result:
            return _stable_id(result["id"])
    memory = getattr(result, "memory", None)
    if memory is not None:
        return _result_id(memory)
    if hasattr(result, "id"):
        return _stable_id(result.id)
    return _stable_id(result)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {k!r}")


def _binary_ndcg(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute binary nDCG@K while counting a duplicate ID as one item."""
    top_ids = ranked_ids[:k]
    seen: set[str] = set()
    dcg = 0.0
    for rank, item_id in enumerate(top_ids, start=1):
        relevant = item_id in relevant_ids and item_id not in seen
        if item_id in relevant_ids:
            seen.add(item_id)
        if relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_retrieval(memory: AgentMemory, cases: Iterable[RetrievalCase]) -> EvaluationReport:
    """Evaluate retrieval with Recall@K, HitRate@K, MRR and binary nDCG@K.

    Recall is a macro average over queries with at least one relevant ID.
    Queries with an empty relevant set have undefined Recall (``None`` in
    their detail) and are excluded from that average. HitRate, MRR and nDCG
    are defined as 0.0 for such a query because no result can be relevant.
    """
    details: list[dict[str, Any]] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    hit_rates: list[float] = []
    ndcgs: list[float] = []

    for case in cases:
        _validate_k(case.top_k)
        results = memory.retrieve(case.query, user_id=case.user_id,
                                  session_id=case.session_id, top_k=case.top_k)
        ranked_ids = [_result_id(result) for result in list(results)[:case.top_k]]
        relevant_ids = {_stable_id(item_id) for item_id in case.relevant_memory_ids}
        unique_hits = {item_id for item_id in ranked_ids if item_id in relevant_ids}
        hit_positions = [rank for rank, item_id in enumerate(ranked_ids, start=1)
                         if item_id in relevant_ids]
        rr = 1.0 / hit_positions[0] if hit_positions else 0.0
        hit_rate = 1.0 if unique_hits else 0.0
        recall = len(unique_hits) / len(relevant_ids) if relevant_ids else None
        ndcg = _binary_ndcg(ranked_ids, relevant_ids, case.top_k)
        if recall is not None:
            recalls.append(recall)
        mrrs.append(rr)
        hit_rates.append(hit_rate)
        ndcgs.append(ndcg)
        details.append({
            "query": case.query,
            "top_k": case.top_k,
            "returned_ids": ranked_ids,
            "relevant_ids": sorted(relevant_ids),
            "unique_hits": sorted(unique_hits),
            "hits": len(unique_hits),
            "relevant_count": len(relevant_ids),
            "recall@k": recall,
            "hit_rate@k": hit_rate,
            "rr": rr,
            "ndcg@k": ndcg,
            "recall_defined": recall is not None,
        })

    query_count = len(details)
    recall_query_count = len(recalls)
    metrics: dict[str, Any] = {
        "recall@k": sum(recalls) / recall_query_count if recalls else None,
        "hit_rate@k": sum(hit_rates) / query_count if query_count else 0.0,
        "mrr": sum(mrrs) / query_count if query_count else 0.0,
        "ndcg@k": sum(ndcgs) / query_count if query_count else 0.0,
        "query_count": query_count,
        "recall_defined_query_count": recall_query_count,
    }
    return EvaluationReport(metrics, details)


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
