from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterable

from .models import Memory, MemoryDecision, MemoryTier, SearchResult, utc_now
from .policy import HeuristicMemoryPolicy, MemoryPolicy
from .stores import InMemoryStore, MemoryStore


class AgentMemory:
    """Facade coordinating construction, tiered storage and cross-store retrieval."""

    def __init__(self, *, policy: MemoryPolicy | None = None,
                 stores: dict[MemoryTier, MemoryStore] | None = None):
        self.policy = policy or HeuristicMemoryPolicy()
        self.stores = stores or {
            MemoryTier.SHORT_TERM: InMemoryStore("short_term", MemoryTier.SHORT_TERM),
            MemoryTier.LONG_TERM: InMemoryStore("long_term", MemoryTier.LONG_TERM),
        }

    def construct(self, content: str, *, user_id: str, session_id: str | None = None,
                  metadata: dict[str, Any] | None = None, tags: list[str] | None = None,
                  importance: float | None = None) -> tuple[MemoryDecision, Memory | None]:
        metadata = dict(metadata or {})
        decision = self.policy.decide(content, user_id=user_id, session_id=session_id, metadata=metadata)
        if not decision.should_remember or decision.tier is None:
            return decision, None
        now = utc_now()
        expires = now + timedelta(seconds=decision.ttl_seconds) if decision.ttl_seconds else None
        memory = Memory(content=content, user_id=user_id, session_id=session_id, tier=decision.tier,
                        importance=importance if importance is not None else decision.score,
                        confidence=metadata.pop("confidence", 1.0), tags=tags or [], metadata=metadata,
                        expires_at=expires)
        self.stores[decision.tier].put(memory)
        return decision, memory

    def retrieve(self, query: str, *, user_id: str, session_id: str | None = None,
                 top_k: int = 5, tiers: Iterable[MemoryTier] | None = None,
                 filters: dict | None = None) -> list[SearchResult]:
        selected = list(tiers) if tiers is not None else list(self.stores)
        results: list[SearchResult] = []
        for tier in selected:
            results.extend(self.stores[tier].search(query, user_id=user_id, session_id=session_id,
                                                    top_k=top_k, filters=filters))
        results.sort(key=lambda item: item.score, reverse=True)
        for result in results[:top_k]:
            result.memory.last_accessed_at = utc_now()
            result.memory.access_count += 1
        return results[:top_k]

    def context(self, query: str, **kwargs: Any) -> str:
        return "\n".join(f"[{r.memory.tier.value}] {r.memory.content}" for r in self.retrieve(query, **kwargs))

    def forget(self, memory_id: str, *, tier: MemoryTier | None = None) -> bool:
        tiers = [tier] if tier else list(self.stores)
        return any(self.stores[t].delete(memory_id) for t in tiers)

    def clear_expired(self) -> int:
        return sum(store.clear_expired() for store in self.stores.values())
