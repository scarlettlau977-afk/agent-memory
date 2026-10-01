from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MemoryTier(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MemoryDecision:
    should_remember: bool
    tier: MemoryTier | None
    score: float
    rationale: str = ""
    ttl_seconds: int | None = None


@dataclass
class Memory:
    content: str
    user_id: str
    session_id: str | None = None
    tier: MemoryTier = MemoryTier.SHORT_TERM
    importance: float = 0.5
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    last_accessed_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    access_count: int = 0

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.expires_at is not None and self.expires_at <= (now or utc_now())


@dataclass
class SearchResult:
    memory: Memory
    score: float
    source: str
    highlights: list[str] = field(default_factory=list)
