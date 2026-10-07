from agent_memory import (
    ConstructionCandidate, ConstructionExperiment, ExplicitOnlyPolicy,
    QualityAwarePolicy, StoreAllPolicy, estimate_tokens,
)
from agent_memory.construction import evaluate_decisions


def candidate(text, **kwargs):
    return ConstructionCandidate("x", "c", "u", text, **kwargs)


def test_store_all_and_explicit_only_semantics():
    items = [candidate("I prefer Python"), candidate("Please remember that I use Python")]
    result = ConstructionExperiment(items, [StoreAllPolicy(), ExplicitOnlyPolicy()]).run()
    assert result["strategies"]["store_all"]["metrics"]["saved_count"] == 2
    assert result["strategies"]["explicit_only"]["metrics"]["saved_count"] == 1


def test_quality_score_is_deterministic_and_duplicate_is_rejected():
    items = [candidate("I prefer Python", should_store=True), candidate("I prefer Python", should_store=False, duplicate_of="x")]
    first = ConstructionExperiment(items, [QualityAwarePolicy()]).run()
    second = ConstructionExperiment(items, [QualityAwarePolicy()]).run()
    a = first["strategies"]["quality_aware"]
    assert a["metrics"]["duplicate_count"] == 1
    assert a["decisions"][1].predicted_should_store is False
    assert first["strategies"]["quality_aware"]["metrics"] == second["strategies"]["quality_aware"]["metrics"]
    assert [d.stored_memory_id for d in a["decisions"]] == [
        d.stored_memory_id for d in second["strategies"]["quality_aware"]["decisions"]
    ]


def test_duplicate_check_does_not_cross_user_boundaries():
    items = [
        ConstructionCandidate("a", "c", "user-a", "I prefer Python", should_store=True),
        ConstructionCandidate("b", "c", "user-b", "I prefer Python", should_store=True),
    ]
    decisions = ConstructionExperiment(items, [QualityAwarePolicy()]).run()["strategies"]["quality_aware"]["decisions"]
    assert all(decision.predicted_should_store for decision in decisions)
    assert not decisions[1].duplicate_detected


def test_labels_are_not_counted_when_unannotated_and_tokens_are_stable():
    items = [candidate("hello", should_store=None), candidate("中文 Python")]
    result = ConstructionExperiment(items, [StoreAllPolicy()]).run()["strategies"]["store_all"]["metrics"]
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (0, 0, 0, 0)
    assert estimate_tokens("中文 Python") == 3


def test_update_is_preserved_by_quality_policy():
    items = [candidate("I prefer Python", should_store=True), candidate("I now prefer Java", should_store=True, is_update_or_conflict=True)]
    result = ConstructionExperiment(items, [QualityAwarePolicy()]).run()["strategies"]["quality_aware"]["metrics"]
    assert result["update_preservation_rate"] == 1.0


def test_routing_denominator_excludes_rejected_expected_items():
    items = [candidate("temporary", should_store=True, target_memory_type="short_term"), candidate("noise", should_store=True, target_memory_type="long_term")]
    decisions = [
        type("D", (), {"predicted_should_store": True, "predicted_memory_type": "short_term", "duplicate_detected": False, "estimated_tokens": 1})(),
        type("D", (), {"predicted_should_store": False, "predicted_memory_type": None, "duplicate_detected": False, "estimated_tokens": 1})(),
    ]
    metrics = evaluate_decisions(items, decisions, store_all_saved=2)
    assert metrics["routing_accuracy"] == 0.5
    assert metrics["routing_coverage"] == 0.5
