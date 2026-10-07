from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .models import Memory, MemoryTier
from .policy import HeuristicMemoryPolicy
from .stores import InMemoryStore


def estimate_tokens(text: str) -> int:
    """Estimate token count consistently without relying on a model tokenizer.

    ASCII words and numbers count as one token each. CJK characters are counted
    individually. This is a comparison-friendly estimate, not billing data.
    """
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text))


@dataclass(frozen=True)
class ConstructionCandidate:
    candidate_id: str
    conversation_id: str
    user_id: str
    text: str
    should_store: bool | None = None
    target_memory_type: str | None = None
    explicit_memory_request: bool = False
    utility_label: float | None = None
    stability_label: float | None = None
    duplicate_of: str | None = None
    is_update_or_conflict: bool = False
    expected_reason: str = ""
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstructionDecision:
    strategy: str
    candidate_id: str
    conversation_id: str
    predicted_should_store: bool
    predicted_memory_type: str | None
    rationale: str
    score: float | None
    stored_memory_id: str | None
    stored_memory: dict[str, Any] | None
    estimated_tokens: int
    duplicate_detected: bool
    is_update_or_conflict: bool


DecisionTuple = tuple[bool, MemoryTier | None, float | None, str]


class ConstructionPolicy(ABC):
    name: str

    @abstractmethod
    def decide(
        self,
        candidate: ConstructionCandidate,
        *,
        store: InMemoryStore,
    ) -> DecisionTuple:
        """Decide whether a candidate should be stored and in which tier."""
        ...


def _fallback_tier(candidate: ConstructionCandidate) -> MemoryTier:
    text = candidate.text.lower()
    durable_markers = ("remember", "记住", "preference", "偏好", "我喜欢", "我叫")
    if candidate.explicit_memory_request or any(marker in text for marker in durable_markers):
        return MemoryTier.LONG_TERM
    return MemoryTier.SHORT_TERM


def _duplicate(candidate: ConstructionCandidate, store: InMemoryStore) -> bool:
    normalized = re.sub(r"\s+", "", candidate.text).lower()
    return any(
        item.user_id == candidate.user_id
        and re.sub(r"\s+", "", item.content).lower() == normalized
        for item in store.all()
    )


class StoreAllPolicy(ConstructionPolicy):
    name = "store_all"

    def decide(self, candidate: ConstructionCandidate, *, store: InMemoryStore) -> DecisionTuple:
        if not candidate.text.strip():
            return False, None, 0.0, "empty candidate"
        return True, _fallback_tier(candidate), 1.0, "store every non-empty candidate"


class HeuristicConstructionPolicy(ConstructionPolicy):
    name = "heuristic"

    def __init__(self):
        self._policy = HeuristicMemoryPolicy()

    def decide(self, candidate: ConstructionCandidate, *, store: InMemoryStore) -> DecisionTuple:
        decision = self._policy.decide(candidate.text, user_id=candidate.user_id,
                                       session_id=candidate.session_id, metadata=candidate.metadata)
        return decision.should_remember, decision.tier, decision.score, decision.rationale


class ExplicitOnlyPolicy(ConstructionPolicy):
    name = "explicit_only"

    _markers = ("请记住", "请牢记", "以后记住", "remember that", "please remember", "remember my")

    def decide(self, candidate: ConstructionCandidate, *, store: InMemoryStore) -> DecisionTuple:
        text = candidate.text.strip().lower()
        explicit = candidate.explicit_memory_request or any(marker in text for marker in self._markers)
        if explicit:
            return True, MemoryTier.LONG_TERM, 1.0, "explicit memory request"
        return False, None, 0.0, "no explicit memory request"


@dataclass(frozen=True)
class QualityConfig:
    utility_weight: float = 0.4
    stability_weight: float = 0.3
    explicit_weight: float = 0.2
    redundancy_weight: float = 0.1
    threshold: float = 0.5
    long_term_threshold: float = 0.72


class QualityAwarePolicy(ConstructionPolicy):
    name = "quality_aware"

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()

    UTILITY_MARKERS = ("我叫", "我的", "我喜欢", "偏好", "目标", "prefer", "my name")
    TEMPORARY_MARKERS = ("今天", "当前", "这次", "暂时", "临时", "for now", "today")

    @staticmethod
    def _utility(text: str) -> float:
        lowered = text.lower()
        if any(marker in lowered for marker in QualityAwarePolicy.UTILITY_MARKERS):
            return 0.9
        return 0.65 if len(text) >= 16 else 0.25

    @staticmethod
    def _stability(text: str) -> float:
        lowered = text.lower()
        if any(marker in lowered for marker in QualityAwarePolicy.TEMPORARY_MARKERS):
            return 0.15
        return 0.8

    def decide(self, candidate: ConstructionCandidate, *, store: InMemoryStore) -> DecisionTuple:
        text = candidate.text.strip()
        if not text:
            return False, None, 0.0, "empty candidate"
        duplicate = _duplicate(candidate, store)
        utility = self._utility(text)
        stability = self._stability(text)
        explicit = float(
            candidate.explicit_memory_request
            or any(marker in text.lower() for marker in ExplicitOnlyPolicy._markers)
        )
        redundancy = 1.0 if duplicate else 0.0
        c = self.config
        score = c.utility_weight * utility + c.stability_weight * stability + c.explicit_weight * explicit - c.redundancy_weight * redundancy
        # The update/conflict flag is evaluation-only annotation. Detect a
        # likely replacement from the candidate text instead of consulting it.
        update_markers = ("no longer", "instead", "now prefer", "改为", "不再", "更新为")
        looks_like_update = any(marker in text.lower() for marker in update_markers)
        if duplicate and not looks_like_update:
            return False, None, round(score, 6), "duplicate candidate rejected"
        if score < c.threshold:
            return False, None, round(score, 6), "quality score below threshold"
        tier = MemoryTier.LONG_TERM if score >= c.long_term_threshold or explicit else MemoryTier.SHORT_TERM
        return True, tier, round(score, 6), "quality score passed threshold"


def _make_memory(candidate: ConstructionCandidate, tier: MemoryTier, score: float | None, strategy: str) -> Memory:
    metadata = dict(candidate.metadata)
    metadata.update({"candidate_id": candidate.candidate_id, "strategy": strategy, "conversation_id": candidate.conversation_id})
    return Memory(content=candidate.text, user_id=candidate.user_id, session_id=candidate.session_id,
                  tier=tier, importance=score if score is not None else 0.5, metadata=metadata,
                  id=f"{strategy}:{candidate.candidate_id}")


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def evaluate_decisions(
    candidates: list[ConstructionCandidate],
    decisions: list[ConstructionDecision],
    *,
    store_all_saved: int,
) -> dict[str, Any]:
    """Calculate storage, decision, routing, and token-cost metrics."""
    if len(candidates) != len(decisions):
        raise ValueError("candidates and decisions must have the same length")

    tp = fp = fn = tn = 0
    eligible_routing = routing_correct = 0
    expected_saved = sum(c.should_store is True for c in candidates)
    unnecessary = missed = 0
    for c, d in zip(candidates, decisions):
        if c.should_store is not None:
            if d.predicted_should_store and c.should_store: tp += 1
            elif d.predicted_should_store: fp += 1
            elif c.should_store: fn += 1
            else: tn += 1
        if c.should_store is True:
            if not d.predicted_should_store: missed += 1
        if c.should_store is False and d.predicted_should_store:
            unnecessary += 1
        if c.should_store is True and c.target_memory_type in {"short_term", "long_term"}:
            eligible_routing += 1
            routing_correct += int(d.predicted_should_store and d.predicted_memory_type == c.target_memory_type)
    saved = sum(d.predicted_should_store for d in decisions)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    duplicate_count = sum(d.duplicate_detected for d in decisions)
    updates = sum(c.is_update_or_conflict for c in candidates)
    preserved_updates = sum(c.is_update_or_conflict and d.predicted_should_store for c, d in zip(candidates, decisions))
    long_expected = sum(c.should_store is True and c.target_memory_type == "long_term" for c in candidates)
    long_saved = sum(d.predicted_should_store and d.predicted_memory_type == "long_term" for d in decisions)
    long_tp = sum(
        c.should_store is True
        and c.target_memory_type == "long_term"
        and d.predicted_should_store
        and d.predicted_memory_type == "long_term"
        for c, d in zip(candidates, decisions)
    )
    tokens = sum(getattr(d, "estimated_tokens", 0) for d in decisions if d.predicted_should_store)
    return {"candidate_count": len(candidates), "saved_count": saved, "rejected_count": len(candidates) - saved,
            "expected_saved_count": expected_saved, "unnecessary_saved_count": unnecessary, "missed_necessary_count": missed,
            "short_term_count": sum(d.predicted_memory_type == "short_term" for d in decisions),
            "long_term_count": long_saved, "duplicate_count": duplicate_count,
            "duplicate_rejection_rate": _safe_div(sum(d.duplicate_detected and not d.predicted_should_store for d in decisions), duplicate_count),
            "update_preservation_rate": _safe_div(preserved_updates, updates),
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "routing_accuracy": _safe_div(routing_correct, eligible_routing), "routing_coverage": _safe_div(sum(d.predicted_should_store and c.should_store is True for c, d in zip(candidates, decisions)), eligible_routing),
            "conditional_routing_accuracy": _safe_div(routing_correct, sum(d.predicted_should_store and c.should_store is True for c, d in zip(candidates, decisions))),
            "total_stored_tokens": tokens, "avg_stored_tokens_per_candidate": _safe_div(tokens, len(candidates)),
            "avg_tokens_per_stored_item": _safe_div(tokens, saved), "token_savings_vs_store_all": _safe_div(store_all_saved - saved, store_all_saved),
            "long_term_precision": _safe_div(long_tp, long_saved), "long_term_recall": _safe_div(long_tp, long_expected)}


class ConstructionExperiment:
    def __init__(
        self,
        candidates: Iterable[ConstructionCandidate],
        policies: Iterable[ConstructionPolicy] | None = None,
    ):
        self.candidates = list(candidates)
        self.policies = list(policies) if policies is not None else [
            StoreAllPolicy(),
            HeuristicConstructionPolicy(),
            ExplicitOnlyPolicy(),
            QualityAwarePolicy(),
        ]

    def run(self) -> dict[str, Any]:
        summaries: dict[str, Any] = {}
        store_all_saved = sum(bool(candidate.text.strip()) for candidate in self.candidates)
        for policy in self.policies:
            store = InMemoryStore(policy.name)
            decisions = []
            for c in self.candidates:
                duplicate = _duplicate(c, store)
                save, tier, score, rationale = policy.decide(c, store=store)
                memory = _make_memory(c, tier, score, policy.name) if save and tier else None
                if memory:
                    store.put(memory)
                stored = ({"id": memory.id, "content": memory.content, "user_id": memory.user_id,
                           "session_id": memory.session_id, "tier": memory.tier.value,
                           "metadata": memory.metadata} if memory else None)
                decisions.append(
                    ConstructionDecision(
                        strategy=policy.name,
                        candidate_id=c.candidate_id,
                        conversation_id=c.conversation_id,
                        predicted_should_store=save,
                        predicted_memory_type=tier.value if tier else None,
                        rationale=rationale,
                        score=score,
                        stored_memory_id=memory.id if memory else None,
                        stored_memory=stored,
                        estimated_tokens=estimate_tokens(c.text),
                        duplicate_detected=duplicate,
                        is_update_or_conflict=c.is_update_or_conflict,
                    )
                )
            summaries[policy.name] = {"metrics": evaluate_decisions(self.candidates, decisions, store_all_saved=store_all_saved), "decisions": decisions}
        return {
            "dataset": {
                "candidate_count": len(self.candidates),
                "annotated_count": sum(c.should_store is not None for c in self.candidates),
            },
            "strategies": summaries,
        }


def candidate_from_dict(item: dict[str, Any]) -> ConstructionCandidate:
    return ConstructionCandidate(**item)


def decision_to_dict(decision: ConstructionDecision) -> dict[str, Any]:
    return asdict(decision)
