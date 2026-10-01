from __future__ import annotations

import argparse

from agent.agent import AgentMode, MinimalAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal memory evaluation agent")
    parser.add_argument("--mode", choices=[mode.value for mode in AgentMode], default=AgentMode.NO_MEMORY.value)
    args = parser.parse_args()
    agent = MinimalAgent.from_mode(args.mode)
    print(f"mode={agent.mode.value}; type 'exit' to stop")
    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        print(f"agent> {agent.chat(user_input)}")


if __name__ == "__main__":
    main()
