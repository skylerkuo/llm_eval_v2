from __future__ import annotations

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from agent.config import get_config

_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str | None = None) -> InMemoryChatMessageHistory:
    sid = session_id or get_config().agent.session_id
    if sid not in _store:
        _store[sid] = InMemoryChatMessageHistory()
    return _store[sid]


def format_recent_history(session_id: str | None = None) -> str:
    cfg = get_config()
    history = get_session_history(session_id)
    messages = history.messages[-(cfg.agent.max_history_turns * 2):]
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role = "使用者" if isinstance(msg, HumanMessage) else "助理"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def record_turn(user_input: str, response_text: str, session_id: str | None = None) -> None:
    history = get_session_history(session_id)
    history.add_message(HumanMessage(content=user_input))
    history.add_message(AIMessage(content=response_text))
