# LLM Eval

工業場域語音引導學習系統，使用 **Gemma 4 E4B (GGUF)** 本地模型與現代 Agent 架構。

## 架構

```
使用者輸入 / 語音
       ↓
  AgentOrchestrator
       ↓
  RouterSkill（選知識檔 + 分類 intent）
       ↓
  Skill Handler（answer / clarify / plan / robot_action / modify / env_query）
       ↓
  Robot WebSocket（可選）+ 對話記憶
```

### 目錄

| 路徑 | 說明 |
|------|------|
| `agent/` | Agent 核心：路由、技能、LLM、知識庫、記憶 |
| `agent/skills/` | 各 intent 的 Skill handler |
| `ui/` | Gradio 介面 |
| `integrations/` | 機器人 WebSocket 客戶端 |
| `main_data/` | 問答 / 分類 / 澄清知識 |
| `skill_data/` | 任務分解 SOP |
| `config.yaml` | 模型、路徑、機器人、評測設定 |

## 下載模型

從 Hugging Face 下載 GGUF 模型並放到專案根目錄：

https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF

預設檔名：`gemma-4-E4B-it-Q4_K_M.gguf`（可在 `config.yaml` 修改）

## 安裝

建議 Python 3.10+。`llama-cpp-python` 請安裝 CUDA 版本：

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

```bash
pip install -r requirements.txt
```

## 執行

### UI 對話（含語音）

```bash
python main.py
```

或沿用舊入口：

```bash
python gai_demo_main_voice.py
```

### 語言模型評測

```bash
python eval.py
```

## 設定

編輯 `config.yaml`：

- `model.path` — Gemma E4B GGUF 路徑
- `model.n_ctx` — 上下文長度（建議 4096+）
- `robot.websocket_uri` — 機器人 WebSocket 位址
- `vision.enabled` — 是否啟用多模態視覺（需額外 transformers 模型）

## Intent 類型

| Intent | 說明 |
|--------|------|
| `answer` | 工業知識問答 |
| `clarify` | 指令不清，追問使用者 |
| `plan` | 任務分解並可送機器人執行 |
| `robot_action` | 解析機器人動作 JSON |
| `modify` | 補充 SOP 規則到 skill 檔 |
| `env_query` | 環境視覺查詢 |

## 嵌入其他系統（純同步，無 async）

整個 Agent API 都是同步的，方便整合：

```python
import sys
sys.path.insert(0, "/path/to/llm_eval-main")

from agent.orchestrator import AgentOrchestrator

agent = AgentOrchestrator()  # 啟動時建立一次

result = agent.run(
    "幫我把橘子放到托盤",
    session_id="user1",
    send_robot=False,      # 機器人由你的系統控制時設 False
    record_memory=True,
)

print(result.response_text)           # 回給使用者的文字
print(result.intent)                  # answer / clarify / plan / ...
print(result.payload.get("steps"))    # plan 時的步驟
print(result.payload.get("actions"))  # robot_action 時的動作
```

不需要 `asyncio` / `await`。

## 與舊版差異

- 單一 `AgentOrchestrator` 取代分散的 `*_llm.py`
- `RouterSkill` + `Skill` handler 模式，易於擴充
- `config.yaml` 集中設定
- JSON 修復與 Gemma 輸出清洗統一在 `agent/parser.py`
- 知識庫封裝為 `KnowledgeBase` 類別
- **全程同步**：無 `asyncio` / `async` / `await`
