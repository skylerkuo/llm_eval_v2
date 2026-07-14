from __future__ import annotations

from agent.chain import build_json_chain, invoke_json_chain
from agent.skills.base import BaseSkill
from agent.state import AgentResult, RouterResult


class RobotActionSkill(BaseSkill):
    intent = "robot_action"

    SYSTEM = """你是一個工業場域機器人動作指令解析器。
你只能輸出 JSON。

請根據提供的檔案與使用者指令，輸出要執行的機器人動作，並產生一句要回覆使用者的 answer。

輸出 JSON 格式：

{{
  "type": "robot_action",
  "answer": "(簡短回覆使用者你會完成什麼動作)",
  "actions": [
    {{
      "name": "function_name",
      "args": {{}}
    }}
  ]
}}

記得給使用者answer回覆
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。"""

    HUMAN = """skill.md：

name: {name}
description: {description}

content:
{content}

使用者指令：
{user_text}"""

    def __init__(self) -> None:
        self._chain = build_json_chain(self.SYSTEM, self.HUMAN)

    def run(self, user_input: str, route: RouterResult) -> AgentResult:
        item = route.knowledge_item
        result = invoke_json_chain(self._chain, {
            "name": item["name"],
            "description": item["description"],
            "content": item["content"],
            "user_text": user_input,
        })

        actions = result.get("actions", [])
        extra = self._format_actions(actions, route.selected_name)
        return AgentResult(
            response_text=result.get("answer", ""),
            extra_info=extra,
            intent=self.intent,
            selected_name=route.selected_name,
            payload={"actions": actions, **result},
        )

    @staticmethod
    def _format_actions(actions: list, selected_name: str) -> str:
        if not actions:
            return "### Robot Action\n\n沒有產生可執行的 action。"

        lines = [
            "### Robot Action\n\n",
            f"**Selected skill:** `{selected_name}`\n\n",
        ]
        for i, action in enumerate(actions, start=1):
            name = action.get("name", "")
            args = action.get("args", {})
            lines.append(f"#### Action {i}\n")
            lines.append(f"- **name:** `{name}`\n")
            if isinstance(args, dict) and args:
                lines.append("- **args:**\n")
                for k, v in args.items():
                    lines.append(f"  - `{k}`: `{v}`\n")
            elif isinstance(args, list) and args:
                lines.append("- **args:**\n")
                for idx, v in enumerate(args):
                    lines.append(f"  - `{idx}`: `{v}`\n")
            else:
                lines.append("- **args:** `{}`\n")
        return "".join(lines)
