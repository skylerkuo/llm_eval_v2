from __future__ import annotations

import re
from pathlib import Path

from agent.config import get_config


def parse_front_matter(md_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", md_text, flags=re.DOTALL)
    if not match:
        return result
    for line in match.group(1).splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def remove_front_matter(md_text: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n?", "", md_text, flags=re.DOTALL).strip()


class KnowledgeBase:
    """Markdown knowledge store with lazy reload."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._items: list[dict] | None = None

    def reload(self) -> list[dict]:
        items: list[dict] = []
        if not self.base_dir.exists():
            self._items = items
            return items

        for path in sorted(self.base_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            meta = parse_front_matter(text)
            items.append({
                "filename": path.name,
                "name": meta.get("name", path.stem),
                "description": meta.get("description", ""),
                "content": remove_front_matter(text),
                "path": str(path),
            })
        self._items = items
        return items

    @property
    def items(self) -> list[dict]:
        if self._items is None:
            self.reload()
        return self._items or []

    def catalog(self) -> str:
        return "\n".join(
            f'- "{item["name"]}": {item["description"]}'
            for item in self.items
        )

    def get(self, name: str) -> dict:
        for item in self.items:
            if item["name"] == name:
                return item
        raise ValueError(f"Knowledge item not found: {name}")


_main_kb: KnowledgeBase | None = None
_skill_kb: KnowledgeBase | None = None


def get_main_knowledge() -> KnowledgeBase:
    global _main_kb
    if _main_kb is None:
        _main_kb = KnowledgeBase(get_config().main_data_dir)
    return _main_kb


def get_skill_knowledge() -> KnowledgeBase:
    global _skill_kb
    if _skill_kb is None:
        _skill_kb = KnowledgeBase(get_config().skill_data_dir)
    return _skill_kb
