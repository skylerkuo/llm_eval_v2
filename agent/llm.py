from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_community.chat_models import ChatLlamaCpp

from agent.config import get_config

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_chat_model: BaseChatModel | None = None


def _load_attempts(cfg) -> list[dict]:
    """Ordered load strategies: configured → partial GPU → CPU."""
    attempts = [
        {
            "n_gpu_layers": cfg.model.n_gpu_layers,
            "n_ctx": cfg.model.n_ctx,
            "label": "configured",
        },
    ]
    if cfg.model.auto_fallback:
        if cfg.model.n_gpu_layers != 5:
            attempts.append({"n_gpu_layers": 5, "n_ctx": 1200, "label": "partial-gpu"})
        if cfg.model.n_gpu_layers != 0:
            attempts.append({"n_gpu_layers": 0, "n_ctx": 1200, "label": "cpu"})
    return attempts


def get_chat_model() -> BaseChatModel:
    global _chat_model
    if _chat_model is not None:
        return _chat_model

    cfg = get_config()
    model_path = cfg.model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Gemma model not found: {model_path}\n"
            "Download gemma-4-E4B-it GGUF and update config.yaml model.path"
        )

    errors: list[str] = []
    for attempt in _load_attempts(cfg):
        try:
            _chat_model = ChatLlamaCpp(
                model_path=str(model_path),
                n_gpu_layers=attempt["n_gpu_layers"],
                n_ctx=attempt["n_ctx"],
                f16_kv=True,
                verbose=False,
                temperature=cfg.model.temperature,
                max_tokens=cfg.model.max_tokens,
            )
            print(
                f"[llm] Loaded Gemma E4B ({attempt['label']}): "
                f"n_gpu_layers={attempt['n_gpu_layers']}, n_ctx={attempt['n_ctx']}"
            )
            return _chat_model
        except Exception as exc:
            errors.append(f"{attempt['label']}: {exc}")
            print(f"[llm] load failed ({attempt['label']}): {exc}")

    raise RuntimeError(
        "Could not load Gemma E4B model. Tried:\n" + "\n".join(errors)
    )


def invoke_with_token_budget(messages, max_tokens: int | None = None) -> str:
    """Invoke the chat model with an optional per-call token cap."""
    model = get_chat_model()
    if max_tokens is not None and hasattr(model, "pipeline"):
        model.pipeline._forward_params = {"max_new_tokens": max_tokens}

    response = model.invoke(messages)
    return response.content.strip()
