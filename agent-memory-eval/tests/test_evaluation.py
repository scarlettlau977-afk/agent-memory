from evaluation.evaluator import evaluate_result


def test_evaluation_record_identifies_memory_failure():
    record = evaluate_result(
        question="What is my preference?",
        ground_truth="Python",
        answer="I don't know.",
        retrieved_memories=[],
        available_memories=[],
    )
    assert record["correct"] is False
    assert record["diagnosis"] == "memory_failure"


def test_evaluation_record_identifies_retrieval_failure():
    record = evaluate_result(
        question="What is my preference?",
        ground_truth="Python",
        answer="I don't know.",
        retrieved_memories=[],
        available_memories=[{"user": "I prefer Python to Java.", "assistant": "Got it."}],
    )
    assert record["diagnosis"] == "retrieval_failure"


def test_evaluation_record_identifies_reasoning_failure():
    memories = [{"user": "I prefer Python to Java.", "assistant": "Got it."}]
    record = evaluate_result(
        question="What is my preference?",
        ground_truth="Python",
        answer="Java",
        retrieved_memories=memories,
        available_memories=memories,
    )
    assert record["diagnosis"] == "agent_reasoning_failure"
