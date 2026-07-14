"""
Optional multimodal vision backend (separate from E4B text GGUF).

Enable via config.yaml vision.enabled: true
Requires transformers + a vision-capable Gemma checkpoint.
"""

from __future__ import annotations

import json
import re

from agent.config import get_config
from agent.parser import repair_json_text


def _parse_json(raw: str) -> dict:
    cleaned = repair_json_text(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"answer": cleaned}


def vision_infer(user_text: str, image_path: str | None = None) -> dict:
    from PIL import Image
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    cfg = get_config()
    path = image_path or str(cfg.root / cfg.vision.image_path)

    processor = AutoProcessor.from_pretrained(cfg.vision.model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        cfg.vision.model_name,
        device_map="cuda",
    )

    image = Image.open(path).convert("RGB")
    system_prompt = """你是一個工業場域機器人助手。
觀察圖片，根據使用者問題回答物品的位置與數量，回答需具體且簡潔。
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。
你只能輸出 JSON，不可輸出其他文字。
輸出格式：
{
  "answer": "回答內容"
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"問題：{user_text}"},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=text_prompt,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    raw_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return _parse_json(raw_text)
