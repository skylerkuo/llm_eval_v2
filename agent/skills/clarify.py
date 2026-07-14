from __future__ import annotations

from agent.chain import build_json_chain, invoke_json_chain
from agent.memory import format_recent_history
from agent.skills.base import BaseSkill
from agent.state import AgentResult, RouterResult


class ClarifySkill(BaseSkill):
    intent = "clarify"

    SYSTEM = """你是一個工業場域機器人。
參照知識文件和對話紀錄和現在使用者的指令去釐清使用者的指令因為使用者現在指令不清楚或有問題，可以跟使用者要求更多資訊。
使用者用甚麼語言(英文或中文)，clarify就要用相同語言回答。

輸出 JSON：
{{
  "clarify": "回答使用者的問題，或是要求使用者提供更多資訊或釐清指令,must give a clarify"
}}"""

    HUMAN = """知識文件：

name: {name}
description: {description}

content:
{content}

對話紀錄：
{history}

使用者問題：
{user_text}"""

    def __init__(self) -> None:
        self._chain = build_json_chain(self.SYSTEM, self.HUMAN)

    def run(self, user_input: str, route: RouterResult) -> AgentResult:
        item = route.knowledge_item
        history = format_recent_history() or "（無）"
        result = invoke_json_chain(self._chain, {
            "name": item["name"],
            "description": item["description"],
            "content": item["content"],
            "history": history,
            "user_text": user_input,
        })
        clarify = result.get("clarify", "")
        return AgentResult(
            response_text=f"clarify：{clarify}",
            intent=self.intent,
            selected_name=route.selected_name,
            payload=result,
        )
