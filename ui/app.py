from __future__ import annotations

import gradio as gr

from agent.config import get_config
from agent.orchestrator import AgentOrchestrator

_orchestrator: AgentOrchestrator | None = None
_whisper_model = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        cfg = get_config().whisper
        _whisper_model = WhisperModel(cfg.model, device=cfg.device, compute_type="float16")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    if not audio_path:
        return ""
    cfg = get_config().whisper
    segments, _ = get_whisper().transcribe(
        audio_path,
        language=cfg.language,
        initial_prompt="以下是繁體中文的轉錄內容。",
    )
    return "".join(seg.text for seg in segments)


def predict(user_input: str, history: list) -> tuple[list, str]:
    if not user_input or not user_input.strip():
        return history, "waiting..."

    result = get_orchestrator().run(user_input)
    history = history or []
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": result.response_text})
    return history, result.extra_info or "waiting..."


def build_app() -> gr.Blocks:
    cfg = get_config().ui

    with gr.Blocks(title=cfg.title) as demo:
        gr.Markdown(f"# {cfg.title}")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="communication window", height=400)
                msg = gr.Textbox(
                    placeholder="輸入問題，或用下方麥克風錄音後自動填入...",
                    show_label=False,
                )
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎙️ 點擊開始錄音 / 再點一次停止並自動轉文字",
                )
                with gr.Row():
                    submit = gr.Button("send", variant="primary")
                    clear = gr.Button("clear")

            with gr.Column(scale=2):
                gr.Markdown("### task planning (Steps / Details)")
                info_panel = gr.Markdown("waiting...", elem_id="info_panel")

        audio_input.stop_recording(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[msg],
        )

        submit.click(predict, inputs=[msg, chatbot], outputs=[chatbot, info_panel]).then(
            lambda: "", None, [msg]
        )
        msg.submit(predict, inputs=[msg, chatbot], outputs=[chatbot, info_panel]).then(
            lambda: "", None, [msg]
        )
        clear.click(lambda: ([], "waiting..."), None, [chatbot, info_panel], queue=False)

    return demo


def launch() -> None:
    cfg = get_config().ui
    demo = build_app()
    demo.launch(server_name=cfg.host, server_port=cfg.port, share=cfg.share)
