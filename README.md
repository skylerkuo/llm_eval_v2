# LLM Eval（工業場域語音引導 Agent）

## 系統目的

這是一套**本地部署**的工業場景 AI Agent，用來理解使用者（文字／語音）指令，並完成：

1. **工業知識問答**（工具、安全、設備說明等）
2. **指令不清楚時追問**（clarify）
3. **任務分解**（例如取放物品 → 可執行 steps）
4. **機器人動作解析**（例如 `move_to`）
5. **SOP／技能規則補充**（寫回 `skill_data`）
6. （可選）環境視覺查詢、透過 WebSocket 把步驟送給外部機器人

模型使用本機 **Gemma 4 E4B（GGUF）**，不依賴雲端 LLM API。  
若要把本系統接到更大系統，只要呼叫同步 API：`AgentOrchestrator().run(文字)`。

---

## 整體流程

```
使用者輸入（文字或語音）
        ↓
  AgentOrchestrator.run()
        ↓
  RouterSkill：選 main_data 知識檔 + 分類 intent
        ↓
  對應 Skill：answer / clarify / plan / robot_action / modify / env_query
        ↓
  回傳 AgentResult（文字回覆 + 可選 steps/actions）
        ↓
  （可選）WebSocket 送機器人 + 寫入對話記憶
```

---

## 目錄與檔案說明

### 啟動／示例

| 檔案 | 功能 |
|------|------|
| `main.py` | 啟動 Gradio UI（文字 + 語音） |
| `gai_demo_main_voice.py` | 舊入口，內部轉呼叫 UI |
| `example_call.py` | **最簡單示例**：呼叫 `AgentOrchestrator.run()` 並印出回答 |
| `eval.py` | 讀取 `llm_eval.jsonl` 做批量評測，寫入 `eval_result_*.jsonl` |
| `llm_eval_main.py` | 舊評測入口，轉呼叫 `eval.py` |
| `dataset.py` | 輔助腳本（Hugging Face dataset 資訊），與主 Agent 流程無關 |
| `config.yaml` | 模型路徑、GPU、知識庫、機器人、UI、Whisper 等設定 |
| `requirements.txt` | Python 依賴清單 |

### Agent 核心（`agent/`）

| 檔案 | 功能 |
|------|------|
| `agent/orchestrator.py` | **主入口**：路由 → Skill →（可選）機器人 → 記憶 |
| `agent/config.py` | 讀取 `config.yaml`，提供全域設定 |
| `agent/llm.py` | 載入本機 Gemma E4B（`llama-cpp` / ChatLlamaCpp） |
| `agent/chain.py` | 組裝 Prompt + 模型 + JSON 解析鏈 |
| `agent/parser.py` | 清洗 Gemma 特殊 token、修復小模型 JSON |
| `agent/knowledge.py` | 載入 `main_data` / `skill_data` 的 Markdown 知識庫 |
| `agent/memory.py` | 對話歷史（預設最近 4 輪） |
| `agent/state.py` | 資料結構：`RouterResult`、`AgentResult` |
| `agent/vision_model.py` | （可選）多模態視覺推理，預設關閉 |
| `agent/__init__.py` | 套件匯出（惰性載入） |

### Skills（`agent/skills/`）

| 檔案 | 功能 |
|------|------|
| `router.py` | 選知識檔 + 分類 intent |
| `answer.py` | 依知識庫回答問題 |
| `clarify.py` | 資訊不足時追問使用者 |
| `plan.py` | 從 `skill_data` 選技能並分解成 steps |
| `robot_action.py` | 解析機器人可執行 actions（如 `move_to`） |
| `modify.py` | 依使用者需求追加 SOP 到 skill 檔 |
| `vision.py` | 環境視覺查詢（預設 stub） |
| `base.py` | Skill 抽象基底類別 |

### UI／整合

| 檔案 | 功能 |
|------|------|
| `ui/app.py` | Gradio 聊天介面、麥克風、Whisper 轉文字 |
| `integrations/robot_client.py` | **同步** WebSocket，把 plan steps 送給機器人 |

### 知識與資料（非 Python，但執行必需）

| 路徑 | 功能 |
|------|------|
| `main_data/` | 路由／問答／澄清／分類規則的 Markdown |
| `skill_data/` | 任務分解 SOP（actions / objects / 範例） |
| `llm_eval.jsonl` | 評測題庫 |
| `md_file_tutorial.md` | 如何撰寫知識 Markdown 的教學 |

> **注意：** `*.gguf` 模型權重由 `.gitignore` 排除，需自行下載，不會上傳到 Git。

---

## Intent 類型

| Intent | 何時出現 | 結果怎麼用 |
|--------|----------|------------|
| `answer` | 問題清楚且知識庫有答案 | 看 `result.response_text` |
| `clarify` | 指令不清楚 | 看 `result.response_text`（追問） |
| `plan` | 取放／整理等任務完整 | `response_text` + `payload["steps"]` |
| `robot_action` | 移動等直接動作 | `response_text` + `payload["actions"]` |
| `modify` | 要求改 SOP | 回覆 + 寫入 `skill_data` |
| `env_query` | 問桌上有什麼等 | 視覺回覆（預設占位） |

---

## 環境與安裝

### 1. 系統需求

- Python 3.10+
- NVIDIA GPU（建議，用於 `llama-cpp-python` CUDA）
- 下載模型放到專案根目錄

### 2. 下載模型

https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF

預設檔名：`gemma-4-E4B-it-Q4_K_M.gguf`  
路徑可在 `config.yaml` 的 `model.path` 修改。

### 3. 安裝依賴

```bash
# CUDA 版 llama-cpp-python（依你的 CUDA 調整 cu125）
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125

pip install -r requirements.txt
```

---

## 如何執行

### A. 最簡單：命令列問答（推薦先跑這個）

```bash
cd /path/to/llm_eval-main
python example_call.py
```

會問預設問題並印出 `intent` 與 `response_text`。  
改檔案內的 `question = "..."` 即可換問題。

### B. Gradio UI（文字 + 語音）

```bash
python main.py
```

瀏覽器開啟介面後輸入問題，或用麥克風錄音。

### C. 批量評測

```bash
python eval.py
```

讀 `llm_eval.jsonl`，寫入 `config.yaml` 中設定的輸出檔（預設 `eval_result_E4B.jsonl`）。

### D. 嵌入其他系統（同步 API）

```python
from agent.orchestrator import AgentOrchestrator

agent = AgentOrchestrator()  # 程式啟動時建一次

result = agent.run(
    "幫我把橘子放到托盤",
    session_id="user1",
    send_robot=False,      # 機器人由你的系統控制 → False
    record_memory=True,    # 是否記住對話（預設約 4 輪）
)

print(result.response_text)            # 給使用者的話
print(result.intent)                   # answer / clarify / plan / ...
print(result.payload.get("steps"))     # 僅 plan 有內容
print(result.payload.get("actions"))   # 僅 robot_action 有內容
```

- `send_robot=False`：不要用本專案 WebSocket，自己拿 `steps`/`actions`
- `record_memory=True`：寫入對話記憶
- `session_id`：不同使用者用不同 ID，記憶才分開

**全程同步，不需要 `asyncio` / `await`。**

---

## 重要設定（`config.yaml`）

| 項目 | 說明 |
|------|------|
| `model.path` | GGUF 檔名或路徑 |
| `model.n_gpu_layers` | `-1` = 整模上 GPU（需足夠顯存） |
| `model.n_ctx` | 上下文長度（預設 1200；越大越吃顯存） |
| `agent.max_history_turns` | 對話記憶輪數（預設 4） |
| `robot.enabled` / `websocket_uri` | 是否內建推送步驟到機器人 |
| `vision.enabled` | 是否啟用真實視覺模型 |

---

## 常見問題

**Q: 出現 `n_ctx_seq < n_ctx_train`？**  
A: 提示而已，表示目前 context 小於模型最大長度；可依顯存調高 `n_ctx`。

**Q: GPU OOM？**  
A: 把 `n_gpu_layers` 改小（例如 `5`）或 `0`（純 CPU），或降低 `n_ctx`。
