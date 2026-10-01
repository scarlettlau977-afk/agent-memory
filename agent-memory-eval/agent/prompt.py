from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the latest user message. "
    "Use the supplied conversation context only when it is relevant."
)


def build_messages(user_input: str, *, context: str = "") -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Conversation context:\n{context}"})
    messages.append({"role": "user", "content": user_input})
    return messages
