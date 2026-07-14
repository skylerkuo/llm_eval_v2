"""
Output parsing for Gemma 4 (E4B GGUF via llama-cpp).

Small local models often echo chat templates or emit malformed JSON.
This module strips Gemma tokens and repairs common JSON mistakes.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda


def strip_gemma_output(message) -> AIMessage:
    text = message.content if hasattr(message, "content") else str(message)

    for pattern in (
        r"<\|turn>model\s*(.*?)\s*<turn\|>",
        r"<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)",
    ):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return AIMessage(content=match.group(1).strip())

    cleaned = re.sub(r"<bos>|<eos>|<\|turn>[a-z]*|<turn\|>", "", text).strip()
    return AIMessage(content=cleaned)


strip_gemma = RunnableLambda(strip_gemma_output)
json_parser = JsonOutputParser()


def repair_json_text(text: str) -> str:
    """Fix common small-LLM JSON formatting issues."""
    text = text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    text = text.replace("'", '"')
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text


def extract_json(text: str) -> dict:
    """Parse JSON from model output with Gemma-aware fallbacks."""
    text = strip_gemma_output(text).content
    text = repair_json_text(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found in model output:\n{text}")

    candidate = repair_json_text(match.group(0))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model output:\n{text}") from exc
