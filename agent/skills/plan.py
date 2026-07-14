from __future__ import annotations

from agent.chain import build_json_chain, invoke_json_chain
from agent.knowledge import get_skill_knowledge
from agent.skills.base import BaseSkill
from agent.state import AgentResult, RouterResult


class PlanSkill(BaseSkill):
    intent = "plan"

    SELECT_SYSTEM = """你是任務技能選擇助手。
你只能輸出合法 JSON，不可以輸出任何 JSON 以外的文字。

輸出格式：
{{
  "skill_name": "技能name"
}}"""

    SELECT_HUMAN = """技能清單:
{catalog}

現在任務:
{user_text}

請選一個最相關的skill"""

    DECOMPOSE_SYSTEM = """你是任務分解的機器人。

技能說明:
{skill_content}

注意: 新增規則是使用者後面新增的任務規則，請優先考慮使用者新增的任務規則再考慮範例。

輸出格式：

{{
  "steps": [
    {{ "id":"1" , "action": "first action name" , "object": "first object name" , "position": "first position" }},
    {{ "id":"2" , "action": "second action name" , "object": "second object name" , "position": "second position"}}
  ],
  "answer": "(簡短回覆使用者你會完成什麼任務)"
}}

position如果沒有就輸出no
記得給使用者answer回覆
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。
id: 為每個步驟的編號，從1開始遞增。

只能輸出 JSON。"""

    DECOMPOSE_HUMAN = "{user_text}"

    def __init__(self) -> None:
        self._skill_kb = get_skill_knowledge()
        self._select_chain = build_json_chain(self.SELECT_SYSTEM, self.SELECT_HUMAN)
        self._decompose_chain = build_json_chain(self.DECOMPOSE_SYSTEM, self.DECOMPOSE_HUMAN)

    def run(self, user_input: str, route: RouterResult) -> AgentResult:
        catalog = self._skill_kb.catalog()
        selected = invoke_json_chain(self._select_chain, {
            "catalog": catalog,
            "user_text": user_input,
        })
        skill_name = selected.get("skill_name", "")
        skill = self._skill_kb.get(skill_name)

        result = invoke_json_chain(self._decompose_chain, {
            "skill_content": skill["content"],
            "user_text": user_input,
        })

        steps = result.get("steps", [])
        extra = self._format_steps(steps)
        return AgentResult(
            response_text=result.get("answer", "（無回答）"),
            extra_info=extra,
            intent=self.intent,
            selected_name=skill_name,
            payload={"steps": steps, "skill_name": skill_name, **result},
        )

    @staticmethod
    def _format_steps(steps: list) -> str:
        if not steps:
            return ""
        lines = ["### Task Planning：\n"]
        for step in steps:
            if isinstance(step, dict):
                lines.append(
                    f"\n### Step {step.get('id', '')}\n"
                    f"- **Action**：`{step.get('action', 'no action')}`\n"
                    f"- **Object**：`{step.get('object', 'no object')}`\n"
                    f"- **Position**：`{step.get('position', 'no position')}`\n"
                )
            else:
                lines.append(f"- {step}\n")
        return "".join(lines)
