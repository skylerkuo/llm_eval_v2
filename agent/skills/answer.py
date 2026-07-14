from __future__ import annotations

from agent.chain import build_json_chain, invoke_json_chain
from agent.skills.base import BaseSkill
from agent.state import AgentResult, RouterResult


class AnswerSkill(BaseSkill):
    intent = "answer"

    SYSTEM = """你是一個工業場域機器人。

參照知識文件適時調整回答回覆使用者。
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。

輸出 JSON：
{{
  "answer": "回答"
}}"""

    HUMAN = """知識文件：

name: {name}
description: {description}

content:
{content}

使用者問題：
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
        return AgentResult(
            response_text=result.get("answer", ""),
            intent=self.intent,
            selected_name=route.selected_name,
            payload=result,
        )
