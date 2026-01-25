"""Gradio UI for the Sunmarke RAG assistant.

This module builds the web interface and wires it to the RAG pipeline
and voice transcription service. It contains only UI glue and event
handlers; core logic lives in `rag_pipeline.py` and `services/`.
"""

import gradio as gr
from rag_pipeline import rag_stream
from services.voice_service import transcribe_audio
from pathlib import Path
import os


# --- UI Helpers ---
def lock_input():
    return gr.update(interactive=False, placeholder="Processing voice..."), gr.update(interactive=False)


def unlock_input():
    return gr.update(interactive=True, placeholder="Ask a follow-up question..."), gr.update(interactive=True)


speak_js = """
(history) => {
    if (!history || history.length === 0) return;

    // 1. Get the last message from history
    const lastMsg = history[history.length - 1];
    if (lastMsg.role !== 'assistant') return;

    // 2. Extract text (Gradio 6.0 content can be a string or list of blocks)
    let text = "";
    if (typeof lastMsg.content === 'string') {
        text = lastMsg.content;
    } else if (Array.isArray(lastMsg.content)) {
        text = lastMsg.content
            .filter(block => block.type === 'text')
            .map(block => block.text)
            .join(" ");
    }

    if (!text) return;

    // 3. Browser Speech Synthesis
    window.speechSynthesis.cancel(); // Stop any existing speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1; // Slightly faster for natural feel
    window.speechSynthesis.speak(utterance);
}
"""


async def process_voice_input(audio_path):
    """Transcribes audio and returns it to the textbox."""
    if audio_path is None:
        return ""
    text = await transcribe_audio(audio_path)
    return text


async def chat_wrapper(query, hist_a, hist_b, hist_c):
    if not query or query.strip() == "":
        yield hist_a, hist_b, hist_c
        return
    async for updated_a, updated_b, updated_c in rag_stream(query, hist_a, hist_b, hist_c):
        yield updated_a, updated_b, updated_c


# --- Layout ---
with gr.Blocks() as demo:
    # State for History
    history_a = gr.State([])
    history_b = gr.State([])
    history_c = gr.State([])

    gr.Markdown("## SUNMARKE SCHOOL ASSISTANT")

    with gr.Row():
        # Column A: Deepseek
        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### DEEPSEEK")
            chat_a = gr.Chatbot(label=None, height=450)
            audio_a = gr.Audio(visible=False, autoplay=True)
            speak_a = gr.Button("🔊 Speak Response")
            speak_a.click(fn=None, inputs=[chat_a], outputs=None, js=speak_js)

        # Column B: Kimi
        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### KIMI")
            chat_b = gr.Chatbot(label=None, height=450)
            audio_b = gr.Audio(visible=False, autoplay=True)
            speak_b = gr.Button("🔊 Speak Response")
            speak_b.click(fn=None, inputs=[chat_b], outputs=None, js=speak_js)

        # Column C: Gemini
        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### GEMINI")
            chat_c = gr.Chatbot(label=None, height=450)
            audio_c = gr.Audio(visible=False, autoplay=True)
            speak_c = gr.Button("🔊 Speak Response")
            speak_c.click(fn=None, inputs=[chat_c], outputs=None, js=speak_js)

    # Bottom Fixed Input Bar
    with gr.Row(elem_id="bottom-bar"):
        with gr.Column(scale=4):
            user_input = gr.Textbox(show_label=False, placeholder="Ask something...", container=False)
        with gr.Column(scale=1):
            # Mic Button
            mic_btn = gr.Audio(sources=["microphone"], type="filepath", label="Mic", container=False)
        with gr.Column(scale=1):
            submit_btn = gr.Button("Send", variant="primary")

    # --- Event Logic ---
    # Voice-to-Text: Triggered when user stops recording
    mic_btn.stop_recording(process_voice_input, inputs=[mic_btn], outputs=[user_input])

    # Submission Logic
    submit_click = submit_btn.click(lock_input, outputs=[user_input, submit_btn])\
        .then(chat_wrapper, inputs=[user_input, chat_a, chat_b, chat_c], outputs=[chat_a, chat_b, chat_c])\
        .then(lambda: "", outputs=[user_input])\
        .then(unlock_input, outputs=[user_input, submit_btn])

    user_input.submit(lock_input, outputs=[user_input, submit_btn])\
        .then(chat_wrapper, inputs=[user_input, chat_a, chat_b, chat_c], outputs=[chat_a, chat_b, chat_c])\
        .then(lambda: "", outputs=[user_input])\
        .then(unlock_input, outputs=[user_input, submit_btn])

CSS = Path("assets/styles.css").read_text()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
    server_port=os.getenv("PORT", default=5000),
    css=CSS
    )
