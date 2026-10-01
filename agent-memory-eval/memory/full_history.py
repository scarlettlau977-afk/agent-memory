from __future__ import annotations

from .base import BaseMemory


class FullHistoryMemory(BaseMemory):
    def __init__(self) -> None:
        self._interactions: list = []

    def add(self, interaction) -> None:
        self._interactions.append(interaction)

    def retrieve(self, query, top_k=5) -> list:
        # Full History deliberately does no relevance filtering. It returns the
        # most recent interactions, bounded by top_k for safety.
        if top_k <= 0:
            return []
        return list(self._interactions[-top_k:])

    def clear(self) -> None:
        self._interactions.clear()
