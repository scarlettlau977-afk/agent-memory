from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


class BaseMemory(ABC):
    """Unified memory interface used by every Agent memory implementation."""

    @abstractmethod
    def add(self, interaction):
        """Store a new interaction."""
        pass

    @abstractmethod
    def retrieve(self, query, top_k=5):
        """Retrieve relevant memories."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all memories."""
        pass


# Backwards-compatible name for code written against the initial scaffold.
MemoryBackend = BaseMemory
