from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from threading import RLock
from typing import Iterable

from .models import Memory, MemoryTier, SearchResult, utc_now


class MemoryStore(ABC):
    """Storage adapter. Implement this interface for Redis, Postgres or a vector DB."""

    name: str

    @abstractmethod
    def put(self, memory: Memory) -> None: ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool: ...

    @abstractmethod
    def search(self, query: str, *, user_id: str, session_id: str | None = None,
               top_k: int = 5, filters: dict | None = None) -> list[SearchResult]: ...

    @abstractmethod
    def all(self) -> Iterable[Memory]: ...

    def clear_expired(self) -> int:
        removed = 0
        for memory in list(self.all()):
            if memory.is_expired():
                removed += int(self.delete(memory.id))
        return removed


class InMemoryStore(MemoryStore):
    """Deterministic lexical store useful for tests and local development."""

    def __init__(self, name: str, tier: MemoryTier | None = None):
        self.name, self.tier = name, tier
        self._items: dict[str, Memory] = {}
        self._lock = RLock()

    def put(self, memory: Memory) -> None:
        if self.tier is not None and memory.tier != self.tier:
            raise ValueError(f"store {self.name!r} accepts {self.tier.value} memories")
        with self._lock:
            self._items[memory.id] = memory

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            return self._items.pop(memory_id, None) is not None

    def all(self) -> Iterable[Memory]:
        with self._lock:
            return list(self._items.values())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        # Keep Latin words and CJK runs, plus CJK bigrams for substring recall.
        tokens: set[str] = set()
        for part in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
            tokens.add(part)
            if re.fullmatch(r"[\u4e00-\u9fff]+", part):
                tokens.update(part[i:i + 2] for i in range(len(part) - 1))
        return tokens

    def search(self, query: str, *, user_id: str, session_id: str | None = None,
               top_k: int = 5, filters: dict | None = None) -> list[SearchResult]:
        q = self._tokens(query)
        filters = filters or {}
        now = utc_now()
        candidates: list[SearchResult] = []
        for item in self.all():
            if item.user_id != user_id or item.is_expired(now):
                continue
            if session_id is not None and item.tier == MemoryTier.SHORT_TERM and item.session_id != session_id:
                continue
            if any(item.metadata.get(k) != v for k, v in filters.items()):
                continue
            words = self._tokens(item.content + " " + " ".join(item.tags))
            lexical = len(q & words) / max(len(q), 1)
            # Exact metadata/tag matches remain useful for structured queries.
            if not lexical and q and not (q & self._tokens(str(item.metadata))):
                continue
            age_days = max((now - item.last_accessed_at).total_seconds() / 86400, 0)
            recency = 1 / (1 + age_days)
            score = 0.65 * lexical + 0.2 * item.importance + 0.1 * item.confidence + 0.05 * recency
            candidates.append(SearchResult(item, round(score, 6), self.name))
        candidates.sort(key=lambda result: result.score, reverse=True)
        return candidates[:top_k]


class ShortTermStore(InMemoryStore):
    def __init__(self, name: str = "short_term"):
        super().__init__(name, MemoryTier.SHORT_TERM)


class LongTermStore(InMemoryStore):
    def __init__(self, name: str = "long_term"):
        super().__init__(name, MemoryTier.LONG_TERM)
