from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the latest user message. "
    "Use the supplied conversation context only when it is relevant."
)


def format_memories(memories) -> str:
    lines = []
    for memory in memories:
        if isinstance(memory, dict):
            lines.append(f"User: {memory.get('user', '')}\nAssistant: {memory.get('assistant', '')}")
        else:
            user = getattr(memory, "user", None)
            assistant = getattr(memory, "assistant", None)
            if user is not None or assistant is not None:
                lines.append(f"User: {user or ''}\nAssistant: {assistant or ''}")
            else:
                lines.append(str(memory))
    return "\n".join(lines)


def build_messages(user_input: str, *, context: str = "") -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Conversation context:\n{context}"})
    messages.append({"role": "user", "content": user_input})
    return messages
