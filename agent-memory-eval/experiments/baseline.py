from __future__ import annotations

import json
import re
from pathlib import Path

from agent.agent import AgentMode, MinimalAgent
from agent.llm import LLM
from evaluation.evaluator import evaluate_result


DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "conversations.json"


class PreferenceProbeLLM(LLM):
    """Deterministic QA test double for checking memory availability.

    This is not a language model and must not be used to claim LLM performance.
    It answers the final question only when the preference fact is present in
    the prompt's conversation context.
    """

    def generate(self, messages) -> str:
        messages = list(messages)
        question = next((message["content"] for message in reversed(messages)
                         if message.get("role") == "user"), "")
        if not ("what programming language" in question.lower() and "prefer" in question.lower()):
            return "Got it."

        context = "\n".join(message.get("content", "") for message in messages
                            if message.get("role") == "system")
        match = re.search(r"\bI prefer ([A-Za-z][A-Za-z0-9+#.-]*) to ", context, re.IGNORECASE)
        return match.group(1) if match else "I don't know."


def load_case() -> dict:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def run(mode: AgentMode | str, case: dict | None = None) -> dict:
    case = case or load_case()
    agent = MinimalAgent.from_mode(mode, llm=PreferenceProbeLLM())
    result = None
    for turn in case["turns"]:
        result = agent.run(turn["user"])
    return result


def run_evaluation(mode: AgentMode | str, case: dict | None = None) -> dict:
    case = case or load_case()
    agent = MinimalAgent.from_mode(mode, llm=PreferenceProbeLLM())
    final_turn = case["turns"][-1]
    for turn in case["turns"][:-1]:
        agent.run(turn["user"])
    available_memories = list(agent.memory.retrieve("", top_k=10_000))
    result = agent.run(final_turn["user"])
    return evaluate_result(
        question=final_turn["user"],
        ground_truth=final_turn["ground_truth"],
        answer=result["answer"],
        retrieved_memories=result["retrieved_memories"],
        available_memories=available_memories,
    )


def main() -> None:
    case = load_case()
    records = []
    for mode in (AgentMode.NO_MEMORY, AgentMode.FULL_HISTORY):
        record = run_evaluation(mode, case)
        records.append({"mode": mode.value, **record})
        print(f"\n=== {mode.value} ===")
        print(json.dumps(records[-1], ensure_ascii=False, indent=2))
    output_path = Path(__file__).resolve().parent / "results" / "baseline_results.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved: {output_path}")


if __name__ == "__main__":
    main()
