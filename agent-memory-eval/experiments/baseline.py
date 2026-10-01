from __future__ import annotations

from agent.agent import AgentMode, MinimalAgent


def run(mode: AgentMode) -> list[str]:
    agent = MinimalAgent.from_mode(mode)
    return [
        agent.chat("My favorite color is blue."),
        agent.chat("What is my favorite color?"),
    ]


if __name__ == "__main__":
    for mode in (AgentMode.NO_MEMORY, AgentMode.FULL_HISTORY):
        print(f"\n=== {mode.value} ===")
        for answer in run(mode):
            print(answer)
