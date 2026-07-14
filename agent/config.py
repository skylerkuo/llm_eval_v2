from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    path: str = "gemma-4-E4B-it-Q4_K_M.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 1200
    temperature: float = 0.0
    max_tokens: int = 300
    auto_fallback: bool = False


@dataclass
class KnowledgeConfig:
    main_dir: str = "main_data"
    skill_dir: str = "skill_data"


@dataclass
class AgentConfig:
    max_history_turns: int = 4
    session_id: str = "default"


@dataclass
class RobotConfig:
    enabled: bool = True
    websocket_uri: str = "ws://140.113.149.93:8765"


@dataclass
class VisionConfig:
    enabled: bool = False
    image_path: str = "received.jpg"
    model_name: str = "google/gemma-4-E2B-it"


@dataclass
class EvalConfig:
    input: str = "llm_eval.jsonl"
    output: str = "eval_result_E4B.jsonl"


@dataclass
class UIConfig:
    title: str = "子計畫三：語音引導學習"
    host: str = "0.0.0.0"
    port: int = 7210
    share: bool = True


@dataclass
class WhisperConfig:
    model: str = "base"
    device: str = "cuda"
    language: str = "zh"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    root: Path = ROOT

    @property
    def model_path(self) -> Path:
        path = Path(self.model.path)
        if path.is_absolute():
            return path
        return self.root / path

    @property
    def main_data_dir(self) -> Path:
        return self.root / self.knowledge.main_dir

    @property
    def skill_data_dir(self) -> Path:
        return self.root / self.knowledge.skill_dir


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else ROOT / "config.yaml"
    if not config_path.exists():
        return AppConfig()

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def section(name: str, cls):
        return cls(**raw.get(name, {}))

    return AppConfig(
        model=section("model", ModelConfig),
        knowledge=section("knowledge", KnowledgeConfig),
        agent=section("agent", AgentConfig),
        robot=section("robot", RobotConfig),
        vision=section("vision", VisionConfig),
        eval=section("eval", EvalConfig),
        ui=section("ui", UIConfig),
        whisper=section("whisper", WhisperConfig),
        root=ROOT,
    )


_CONFIG: AppConfig | None = None


def get_config() -> AppConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config(os.environ.get("LLM_EVAL_CONFIG"))
    return _CONFIG
