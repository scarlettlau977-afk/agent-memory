from __future__ import annotations

from .base import BaseMemory


class NoMemory(BaseMemory):
    def add(self, interaction) -> None:
        return None

    def retrieve(self, query, top_k=5) -> list:
        return []

    def clear(self) -> None:
        return None
