from experiments.baseline import load_case, run, run_evaluation


def test_first_memory_experiment_matches_expected_answers():
    case = load_case()

    no_memory = run("no_memory", case)
    full_history = run("full_history", case)
    expected = case["turns"][-1]["expected"]

    assert no_memory["answer"] == expected["no_memory"]
    assert no_memory["retrieved_memories"] == []
    assert full_history["answer"] == expected["full_history"]
    assert len(full_history["retrieved_memories"]) == 3


def test_first_memory_experiment_writes_evaluation_fields():
    case = load_case()
    no_memory = run_evaluation("no_memory", case)
    full_history = run_evaluation("full_history", case)

    assert no_memory["diagnosis"] == "memory_failure"
    assert full_history["correct"] is True
    assert full_history["diagnosis"] == "correct"
