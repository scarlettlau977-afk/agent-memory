from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .llm import EchoLLM, LLM
from .prompt import build_messages
from memory.base import MemoryBackend, Turn
from memory.full_history import FullHistoryMemory
from memory.no_memory import NoMemory


class AgentMode(str, Enum):
    NO_MEMORY = "no_memory"
    FULL_HISTORY = "full_history"


@dataclass
class AgentConfig:
    mode: AgentMode = AgentMode.NO_MEMORY


class MinimalAgent:
    """One agent implementation shared by all experiments."""

    def __init__(self, *, llm: LLM | None = None, memory: MemoryBackend | None = None,
                 config: AgentConfig | None = None):
        self.llm = llm or EchoLLM()
        self.config = config or AgentConfig()
        self.memory = memory or self._memory_for_mode(self.config.mode)

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
        interactions = self.memory.retrieve(user_input, top_k=10_000)
        context = self._format_context(interactions)
        response = self.llm.generate(build_messages(user_input, context=context))
        self.memory.add(Turn(user=user_input, assistant=response))
        return response

    @staticmethod
    def _format_context(interactions: list) -> str:
        lines = []
        for interaction in interactions:
            if isinstance(interaction, Turn):
                lines.append(f"User: {interaction.user}\nAssistant: {interaction.assistant}")
            elif isinstance(interaction, dict):
                lines.append(f"User: {interaction.get('user', '')}\nAssistant: {interaction.get('assistant', '')}")
            else:
                lines.append(str(interaction))
        return "\n".join(lines)

    def reset(self) -> None:
        self.memory.clear()

    def transcript(self) -> list[Turn]:
        return self.memory.retrieve("", top_k=10_000)
