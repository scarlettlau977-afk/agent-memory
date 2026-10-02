from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Iterable

from .prompt import build_messages, format_memories


class LLM(ABC):
    """Small provider-neutral LLM interface used by the agent."""

    def build_prompt(self, *, user_message: str, memories) -> list[dict[str, str]]:
        """Build the provider-neutral prompt consumed by ``generate``."""
        return build_messages(user_message, context=format_memories(memories))

    @abstractmethod
    def generate(self, messages: Iterable[dict[str, str]]) -> str:
        raise NotImplementedError


class EchoLLM(LLM):
    """Deterministic local model for smoke tests and development.

    It intentionally exposes whether context was passed, which makes the two
    memory modes easy to inspect before connecting a real model.
    """

    def generate(self, messages: Iterable[dict[str, str]]) -> str:
        messages = list(messages)
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        latest = user_messages[-1] if user_messages else ""
        context_message = next((m["content"] for m in messages
                                if m.get("role") == "system" and m.get("content", "").startswith("Conversation context:\n")), "")
        prior = context_message.count("User: ")
        return f"[echo] {latest} (prior_user_turns={prior})"


class OpenAICompatibleLLM(LLM):
    """Optional OpenAI-compatible adapter.

    The dependency is imported lazily, so the local EchoLLM works without
    installing an SDK or configuring credentials.
    """

    def __init__(self, *, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, **client_kwargs: Any):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional OpenAI client with: pip install openai") from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                             base_url=base_url or os.getenv("OPENAI_BASE_URL"), **client_kwargs)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, messages: Iterable[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=list(messages))
        return response.choices[0].message.content or ""
