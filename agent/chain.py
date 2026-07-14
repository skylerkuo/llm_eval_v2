from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from agent.llm import get_chat_model
from agent.parser import extract_json, json_parser, strip_gemma


@dataclass
class JsonChain:
    full: Runnable
    raw: Runnable


def build_json_chain(system: str, human: str) -> JsonChain:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    raw = prompt | get_chat_model() | strip_gemma
    return JsonChain(full=raw | json_parser, raw=raw)


def invoke_json_chain(chain: JsonChain, variables: dict) -> dict:
    try:
        result = chain.full.invoke(variables)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    raw = chain.raw.invoke(variables)
    content = raw.content if hasattr(raw, "content") else str(raw)
    return extract_json(content)
