from __future__ import annotations

from agent.chain import build_json_chain, invoke_json_chain
from agent.knowledge import get_main_knowledge
from agent.state import RouterResult


class RouterSkill:
    """Two-step router: select knowledge doc, then classify intent."""

    SELECT_SYSTEM = """你是一個機器人檢索助手。
只能輸出 JSON，要找出核問題最相關的檔案，讓後續機器人能夠跟著檔案內容回答問題、釐清、行動、任務規劃、修改SOP。

{{
  "selected_name": "文件name"
}}"""

    SELECT_HUMAN = """知識文件：
{catalog}

問題：
{user_text}

請選最相關文件"""

    CLASSIFY_SYSTEM = """你是一個工業場域機器人。
根據提供的知識和現在使用者的指令選擇是哪一個種類，種類有 plan or env_query or clarify or answer or modify or robot_action。

輸出 JSON：
{{
  "type": "plan or env_query or clarify or answer or modify or robot_action"
}}

只能輸出 JSON。"""

    CLASSIFY_HUMAN = """知識文件：

name: {name}
description: {description}

content:
{content}

問題：
{user_text}"""

    def __init__(self) -> None:
        self._kb = get_main_knowledge()
        self._select_chain = build_json_chain(self.SELECT_SYSTEM, self.SELECT_HUMAN)
        self._classify_chain = build_json_chain(self.CLASSIFY_SYSTEM, self.CLASSIFY_HUMAN)

    def route(self, user_input: str) -> RouterResult:
        catalog = self._kb.catalog()
        selected = invoke_json_chain(self._select_chain, {
            "catalog": catalog,
            "user_text": user_input,
        })
        selected_name = selected.get("selected_name", "")
        item = self._kb.get(selected_name)

        classified = invoke_json_chain(self._classify_chain, {
            "name": item["name"],
            "description": item["description"],
            "content": item["content"],
            "user_text": user_input,
        })

        intent = classified.get("type", "answer")
        print(f"[router] doc={selected_name} intent={intent}")
        return RouterResult(
            intent=intent,
            selected_name=selected_name,
            knowledge_item=item,
        )
