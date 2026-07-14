"""Run JSONL benchmark against the agent orchestrator (synchronous)."""

from __future__ import annotations

import json

from agent.config import get_config
from agent.orchestrator import AgentOrchestrator


def run_eval(input_path: str | None = None, output_path: str | None = None) -> None:
    cfg = get_config()
    input_path = input_path or str(cfg.root / cfg.eval.input)
    output_path = output_path or str(cfg.root / cfg.eval.output)

    orchestrator = AgentOrchestrator()

    with open(input_path, encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_id, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            question = data.get("question", "")
            correct_answer = data.get("answer", "")
            print(f"[{line_id}] {question}")

            record = orchestrator.run_eval_line(question, correct_answer)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    run_eval()
