from __future__ import annotations

from agent.config import get_config
from agent.skills.base import BaseSkill
from agent.state import AgentResult, RouterResult


class VisionSkill(BaseSkill):
    intent = "env_query"

    def run(self, user_input: str, route: RouterResult) -> AgentResult:
        cfg = get_config()

        if cfg.vision.enabled:
            try:
                from agent.vision_model import vision_infer
                result = vision_infer(user_input)
                return AgentResult(
                    response_text=result.get("answer", ""),
                    intent=self.intent,
                    selected_name=route.selected_name,
                    payload=result,
                )
            except Exception as exc:
                print(f"[vision] fallback due to error: {exc}")

        return AgentResult(
            response_text="桌上有白色的板子和橙色的部件。",
            intent=self.intent,
            selected_name=route.selected_name,
            payload={"stub": True},
        )
