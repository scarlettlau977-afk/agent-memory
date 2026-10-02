from __future__ import annotations

from .base import BaseMemory


class FullHistoryMemory(BaseMemory):
    def __init__(self):
        self.history = []

    def add(self, interaction):
        self.history.append(interaction)

    def retrieve(self, query, top_k=5):
        # Deliberately ignore query and top_k: this baseline returns everything.
        return self.history

    def clear(self):
        self.history = []
