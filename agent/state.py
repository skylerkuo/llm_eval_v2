from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


INTENT_TYPES = frozenset({
    "answer",
    "clarify",
    "plan",
    "robot_action",
    "modify",
    "env_query",
})


@dataclass
class RouterResult:
    intent: str
    selected_name: str
    knowledge_item: dict


@dataclass
class AgentResult:
    response_text: str
    extra_info: str = ""
    intent: str = ""
    selected_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
