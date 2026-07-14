from __future__ import annotations

from abc import ABC, abstractmethod

from agent.state import AgentResult, RouterResult


class BaseSkill(ABC):
    intent: str = ""

    @abstractmethod
    def run(self, user_input: str, route: RouterResult) -> AgentResult:
        raise NotImplementedError
