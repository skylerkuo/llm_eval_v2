from __future__ import annotations

from agent.chain import build_json_chain, invoke_json_chain
from agent.knowledge import get_skill_knowledge
from agent.skills.base import BaseSkill
from agent.state import AgentResult, RouterResult


class ModifySkill(BaseSkill):
    intent = "modify"

    SELECT_SYSTEM = """你是一個工業場域知識檢索助手。
你只能輸出 JSON。

{{
  "selected_name": "文件name"
}}"""

    SELECT_HUMAN = """知識文件：
{catalog}

問題：
{user_text}

請選最相關文件"""

    MODIFY_SYSTEM = """你是要負責任務規則補充的機器人。

你需要根據使用者的指令補充任務的描述，讓任務描述更完整。
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。

輸出格式：
{{
  "add": "寫上使用者需要補充的規則或SOP，50字以內",
  "answer": "簡短回覆使用者你會補充什麼"
}}

只能輸出 JSON。"""

    MODIFY_HUMAN = """知識文件：

name: {name}
description: {description}

content:
{content}

使用者問題：
{user_text}"""

    def __init__(self) -> None:
        self._skill_kb = get_skill_knowledge()
        self._select_chain = build_json_chain(self.SELECT_SYSTEM, self.SELECT_HUMAN)
        self._modify_chain = build_json_chain(self.MODIFY_SYSTEM, self.MODIFY_HUMAN)

    def run(self, user_input: str, route: RouterResult) -> AgentResult:
        catalog = self._skill_kb.catalog()
        selected = invoke_json_chain(self._select_chain, {
            "catalog": catalog,
            "user_text": user_input,
        })
        name = selected.get("selected_name", "")
        item = self._skill_kb.get(name)

        result = invoke_json_chain(self._modify_chain, {
            "name": item["name"],
            "description": item["description"],
            "content": item["content"],
            "user_text": user_input,
        })

        add_content = result.get("add", "")
        extra = f"### new rules：\n{add_content}" if add_content else ""

        if add_content:
            with open(item["path"], "a", encoding="utf-8") as f:
                f.write(f"\n- {add_content}\n")
            self._skill_kb.reload()
            print(f"[modify] appended rule to {item['path']}")

        return AgentResult(
            response_text=result.get("answer", ""),
            extra_info=extra,
            intent=self.intent,
            selected_name=name,
            payload=result,
        )
