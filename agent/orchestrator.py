from __future__ import annotations

from agent.config import get_config
from agent.memory import record_turn
from agent.skills import (
    AnswerSkill,
    ClarifySkill,
    ModifySkill,
    PlanSkill,
    RobotActionSkill,
    RouterSkill,
    VisionSkill,
)
from agent.skills.base import BaseSkill
from agent.state import AgentResult


class AgentOrchestrator:
    """
    Sync agent entry point:
      Router → Skill handler → Optional robot dispatch → Memory
    """

    def __init__(self) -> None:
        self.router = RouterSkill()
        self._skills: dict[str, BaseSkill] = {
            skill.intent: skill
            for skill in (
                AnswerSkill(),
                ClarifySkill(),
                PlanSkill(),
                RobotActionSkill(),
                ModifySkill(),
                VisionSkill(),
            )
        }

    def run(
        self,
        user_input: str,
        *,
        session_id: str | None = None,
        send_robot: bool = True,
        record_memory: bool = True,
    ) -> AgentResult:
        user_input = user_input.strip()
        if not user_input:
            return AgentResult(response_text="waiting...", extra_info="waiting...")

        try:
            route = self.router.route(user_input)
            skill = self._skills.get(route.intent)
            if skill is None:
                result = AgentResult(
                    response_text="no class",
                    intent=route.intent,
                    selected_name=route.selected_name,
                )
            else:
                result = skill.run(user_input, route)

            if send_robot:
                self._maybe_dispatch_robot(result)

            if record_memory:
                record_turn(user_input, result.response_text, session_id)

            return result

        except Exception as exc:
            print(f"[orchestrator] error: {exc}")
            return AgentResult(
                response_text="",
                intent="error",
                error=str(exc),
            )

    def _maybe_dispatch_robot(self, result: AgentResult) -> None:
        cfg = get_config()
        if not cfg.robot.enabled:
            return

        from integrations.robot_client import send_robot_message

        if result.intent == "plan":
            steps = result.payload.get("steps", [])
            if steps:
                try:
                    send_robot_message(steps, request_image=False)
                except Exception as exc:
                    print(f"[robot] plan dispatch failed: {exc}")

        elif result.intent == "env_query" and cfg.vision.enabled:
            try:
                send_robot_message("", request_image=True)
            except Exception as exc:
                print(f"[robot] vision request failed: {exc}")

    def run_eval_line(
        self,
        question: str,
        correct_answer: str,
        *,
        send_robot: bool = False,
    ) -> dict:
        result = self.run(
            question,
            send_robot=send_robot,
            record_memory=False,
        )
        return {
            "question": question,
            "correct_answer": correct_answer,
            "model_answer": result.response_text,
            "type": result.intent,
            "selected_name": result.selected_name,
            "extra_info": result.extra_info,
            **({"error": result.error} if result.error else {}),
        }
