from agent.agent import MinimalAgent
from memory.base import Turn
from memory.full_history import FullHistoryMemory


def test_full_history_returns_all_interactions_without_filtering():
    memory = FullHistoryMemory()
    first = Turn("first", "answer one")
    second = Turn("second", "answer two")
    memory.add(first)
    memory.add(second)

    assert memory.retrieve("unrelated query", top_k=1) == [first, second]
    assert memory.retrieve("", top_k=0) is memory.history

    memory.clear()
    assert memory.history == []


def test_full_history_agent_receives_previous_turns():
    agent = MinimalAgent.from_mode("full_history")
    agent.chat("My favorite color is blue.")
    response = agent.chat("What is my favorite color?")

    assert "prior_user_turns=1" in response
    assert len(agent.transcript()) == 2
