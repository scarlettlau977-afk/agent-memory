from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .llm import EchoLLM, LLM
from .prompt import format_memories
from memory.base import MemoryBackend, Turn
from memory.full_history import FullHistoryMemory
from memory.no_memory import NoMemory


class AgentMode(str, Enum):
    NO_MEMORY = "no_memory"
    FULL_HISTORY = "full_history"


@dataclass
class AgentConfig:
    mode: AgentMode = AgentMode.NO_MEMORY


class Agent:
    """Memory-independent agent execution loop."""

    def __init__(self, llm: LLM, memory: MemoryBackend):
        self.llm = llm
        self.memory = memory

    def run(self, user_message):
        # Snapshot results before writing the current interaction. This keeps
        # ``retrieved_memories`` faithful even when a backend returns its live list.
        memories = list(self.memory.retrieve(user_message))
        prompt = self.llm.build_prompt(user_message=user_message, memories=memories)
        answer = self.llm.generate(prompt)
        self.memory.add({"user": user_message, "assistant": answer})
        return {"answer": answer, "retrieved_memories": memories}


class MinimalAgent(Agent):
    """Convenience wrapper that selects one of the baseline memory modes."""

    def __init__(self, *, llm: LLM | None = None, memory: MemoryBackend | None = None,
                 config: AgentConfig | None = None):
        self.llm = llm or EchoLLM()
        self.config = config or AgentConfig()
        super().__init__(self.llm, memory or self._memory_for_mode(self.config.mode))

    @staticmethod
    def _memory_for_mode(mode: AgentMode) -> MemoryBackend:
        if mode == AgentMode.FULL_HISTORY:
            return FullHistoryMemory()
        return NoMemory()

    @classmethod
    def from_mode(cls, mode: AgentMode | str, *, llm: LLM | None = None) -> "MinimalAgent":
        return cls(llm=llm, config=AgentConfig(AgentMode(mode)))

    @property
    def mode(self) -> AgentMode:
        return self.config.mode

    def chat(self, user_input: str) -> str:
        return self.run(user_input)["answer"]

    @staticmethod
    def _format_context(interactions: list) -> str:
        return format_memories(interactions)

    def reset(self) -> None:
        self.memory.clear()

    def transcript(self) -> list[Turn]:
        return self.memory.retrieve("", top_k=10_000)
