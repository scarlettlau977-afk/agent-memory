from dataclasses import dataclass
import json

import pytest

from agent_memory import Memory, RetrievalCase, SearchResult, evaluate_retrieval


@dataclass
class FakeMemory:
    results: list

    def retrieve(self, query, *, user_id, session_id, top_k):
        return self.results[:top_k]


def result(item_id):
    return SearchResult(memory=Memory(content=f"memory {item_id}", user_id="eval-user", id=str(item_id)),
                        score=1.0, source="test")


def evaluate(ids, relevant, top_k=3):
    memory = FakeMemory([result(item_id) for item_id in ids])
    return evaluate_retrieval(memory, [RetrievalCase("q", set(relevant), top_k=top_k)])


def test_recall_and_hit_rate_are_distinct_for_partial_hit():
    report = evaluate(["r1", "x", "x2"], {"r1", "r2", "r3", "r4"})
    assert report.metrics["recall@k"] == 0.25
    assert report.metrics["hit_rate@k"] == 1.0


def test_recall_and_hit_rate_for_three_unique_hits():
    report = evaluate(["r1", "r2", "r3"], {"r1", "r2", "r3", "r4"})
    assert report.metrics["recall@k"] == 0.75
    assert report.metrics["hit_rate@k"] == 1.0


def test_no_hit_sets_all_rank_metrics_to_zero():
    report = evaluate(["x1", "x2", "x3"], {"r1", "r2", "r3", "r4"})
    assert report.metrics["recall@k"] == 0.0
    assert report.metrics["hit_rate@k"] == 0.0
    assert report.metrics["mrr"] == 0.0
    assert report.metrics["ndcg@k"] == 0.0


def test_all_relevant_items_hit():
    report = evaluate(["r1", "r2", "r3", "r4"], {"r1", "r2", "r3", "r4"}, top_k=4)
    assert report.metrics["recall@k"] == 1.0
    assert report.metrics["hit_rate@k"] == 1.0


def test_mrr_uses_first_relevant_rank_and_averages_queries():
    cases = [
        RetrievalCase("q1", {"r"}, top_k=3),
        RetrievalCase("q2", {"r"}, top_k=3),
        RetrievalCase("q3", {"r"}, top_k=3),
    ]
    memories = [
        FakeMemory([result("r"), result("x"), result("y")]),
        FakeMemory([result("x"), result("r"), result("y")]),
        FakeMemory([result("x"), result("y"), result("r")]),
    ]
    reports = [evaluate_retrieval(memory, [case]) for memory, case in zip(memories, cases)]
    assert [report.details[0]["rr"] for report in reports] == [1.0, 0.5, 1 / 3]
    assert sum(report.metrics["mrr"] for report in reports) / 3 == pytest.approx((1 + 0.5 + 1 / 3) / 3)


def test_ndcg_is_one_for_ideal_order_and_lower_for_later_relevant_items():
    ideal = evaluate(["r1", "r2", "x"], {"r1", "r2"})
    later = evaluate(["x", "r1", "r2"], {"r1", "r2"})
    assert ideal.metrics["ndcg@k"] == 1.0
    assert later.metrics["ndcg@k"] < ideal.metrics["ndcg@k"]


def test_duplicate_relevant_id_does_not_inflate_recall():
    report = evaluate(["r1", "r1", "x"], {"r1", "r2", "r3", "r4"})
    assert report.metrics["recall@k"] == 0.25
    assert report.metrics["hit_rate@k"] == 1.0
    assert report.details[0]["hits"] == 1


def test_empty_results_and_empty_relevant_set_are_explicit():
    empty_results = evaluate([], {"r1"})
    assert empty_results.metrics["recall@k"] == 0.0
    assert empty_results.metrics["hit_rate@k"] == 0.0
    assert empty_results.metrics["mrr"] == 0.0

    empty_relevant = evaluate(["x"], set())
    assert empty_relevant.metrics["recall@k"] is None
    assert empty_relevant.metrics["recall_defined_query_count"] == 0
    assert empty_relevant.metrics["hit_rate@k"] == 0.0
    assert empty_relevant.metrics["mrr"] == 0.0
    assert empty_relevant.metrics["ndcg@k"] == 0.0
    assert empty_relevant.details[0]["recall_defined"] is False

    both_empty = evaluate([], set())
    assert both_empty.metrics["recall@k"] is None


def test_k_boundaries_and_actual_result_count():
    report = evaluate(["r1"], {"r1", "r2"}, top_k=5)
    assert report.metrics["recall@k"] == 0.5
    assert evaluate(["r1"], {"r1"}, top_k=1).metrics["recall@k"] == 1.0
    with pytest.raises(ValueError, match="positive integer"):
        evaluate(["r1"], {"r1"}, top_k=0)
    with pytest.raises(ValueError, match="positive integer"):
        evaluate(["r1"], {"r1"}, top_k=-1)


def test_numeric_and_string_ids_are_compared_consistently():
    report = evaluate([1], {"1"})
    assert report.metrics["recall@k"] == 1.0


def test_macro_recall_excludes_queries_with_empty_relevant_sets():
    memory = FakeMemory([result("r1"), result("x")])
    report = evaluate_retrieval(memory, [
        RetrievalCase("q1", {"r1", "r2"}, top_k=2),
        RetrievalCase("q2", set(), top_k=2),
    ])
    assert report.metrics["recall@k"] == 0.5
    assert report.metrics["query_count"] == 2
    assert report.metrics["recall_defined_query_count"] == 1
    json.dumps({"metrics": report.metrics, "details": report.details})
