from agent.agent import MinimalAgent
from memory.no_memory import NoMemory


def test_no_memory_never_stores_or_retrieves():
    memory = NoMemory()
    memory.add({"user": "my name is Ada", "assistant": "hello"})

    assert memory.retrieve("name") == []
    assert memory.retrieve("name", top_k=100) == []

    memory.clear()
    assert memory.retrieve("anything") == []


def test_no_memory_is_baseline_zero_for_agent():
    agent = MinimalAgent.from_mode("no_memory")
    agent.chat("My favorite color is blue.")
    agent.chat("What is my favorite color?")

    assert agent.transcript() == []
    assert "prior_user_turns=0" in agent.chat("Do you remember it?")
