from __future__ import annotations

import json
import re
from typing import Any, Iterable


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def contains_fact(memories: Iterable[Any], fact: str) -> bool:
    target = _normalize(fact)
    return bool(target) and any(target in _normalize(_text(memory)) for memory in memories)


def evaluate_result(*, question: str, ground_truth: str, answer: str,
                    retrieved_memories: list[Any], available_memories: list[Any] | None = None) -> dict[str, Any]:
    """Create a JSON-serializable answer record and diagnose an error source.

    ``available_memories`` represents facts stored before the final question.
    It lets the evaluator distinguish a construction/memory failure from a
    retrieval failure. If the fact was retrieved but the answer is wrong, the
    remaining error is attributed to agent reasoning/answer generation.
    """
    correct = _normalize(answer) == _normalize(ground_truth)
    memory_available = contains_fact(available_memories or [], ground_truth)
    retrieved_fact = contains_fact(retrieved_memories, ground_truth)
    if correct:
        diagnosis = "correct"
    elif not memory_available:
        diagnosis = "memory_failure"
    elif not retrieved_fact:
        diagnosis = "retrieval_failure"
    else:
        diagnosis = "agent_reasoning_failure"
    return {
        "question": question,
        "ground_truth": ground_truth,
        "retrieved_memories": retrieved_memories,
        "answer": answer,
        "correct": correct,
        "diagnosis": diagnosis,
    }
