from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import MemoryDecision, MemoryTier


class MemoryPolicy(ABC):
    """Decision plug-in: decide whether input is worth storing and where."""

    @abstractmethod
    def decide(self, content: str, *, user_id: str, session_id: str | None = None,
               metadata: dict[str, Any] | None = None) -> MemoryDecision: ...


class HeuristicMemoryPolicy(MemoryPolicy):
    """A transparent baseline; replace with an LLM or learned classifier in production."""

    LONG_MARKERS = ("remember", "记住", "我的", "我喜欢", "我叫", "偏好", "习惯", "长期")
    SHORT_MARKERS = ("当前", "这次", "暂时", "今天", "this task", "for now", "临时")
    NOISE = ("你好", "谢谢", "hello", "hi", "好的", "ok", "测试")

    def decide(self, content: str, *, user_id: str, session_id: str | None = None,
               metadata: dict[str, Any] | None = None) -> MemoryDecision:
        text = content.strip().lower()
        metadata = metadata or {}
        if not text or any(text == n or text.startswith(n + " ") for n in self.NOISE):
            return MemoryDecision(False, None, 0.05, "greeting/acknowledgement/noise")
        explicit = any(marker in text for marker in self.LONG_MARKERS)
        transient = any(marker in text for marker in self.SHORT_MARKERS)
        if metadata.get("remember") is False:
            return MemoryDecision(False, None, 0.0, "caller explicitly disabled memory")
        if metadata.get("remember") is True:
            explicit = True
        if explicit:
            return MemoryDecision(True, MemoryTier.LONG_TERM, 0.95, "explicit or durable preference/fact")
        if transient or metadata.get("task_scoped"):
            return MemoryDecision(True, MemoryTier.SHORT_TERM, 0.7, "session/task-scoped context", ttl_seconds=metadata.get("ttl_seconds", 86400))
        # Conservative default: only retain reasonably informative inputs.
        if len(text) >= 12:
            return MemoryDecision(True, MemoryTier.SHORT_TERM, 0.52, "informative but not clearly durable", ttl_seconds=metadata.get("ttl_seconds", 86400))
        return MemoryDecision(False, None, 0.2, "low information value")
