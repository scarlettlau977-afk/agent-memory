from agent.agent import Agent
from agent.llm import EchoLLM
from memory.full_history import FullHistoryMemory
from memory.no_memory import NoMemory


def test_agent_run_has_stable_memory_independent_contract():
    no_memory = Agent(EchoLLM(), NoMemory())
    full_history = Agent(EchoLLM(), FullHistoryMemory())

    no_result = no_memory.run("first")
    full_history.run("first")
    full_result = full_history.run("second")

    assert no_result["answer"].startswith("[echo] first")
    assert no_result["retrieved_memories"] == []
    assert len(full_result["retrieved_memories"]) == 1
    assert full_result["retrieved_memories"][0]["user"] == "first"
