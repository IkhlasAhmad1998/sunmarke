"""Simple Gradio UI for RAG demo.

This module creates a three-column view to show responses from
multiple model providers and a bottom input bar. CSS is read
from `assets/styles.css` so styling is separated from code.
"""

from __future__ import annotations

from typing import Tuple
import pathlib
import logging
import gradio as gr

from rag_pipeline import rag


def process_query(query_text: str) -> Tuple[str, str, str, str]:
    """Process a text query and return the input plus three model responses.

    Returns empty strings if the input is blank.
    """
    if not query_text or query_text.strip() == "":
        return "", "", ""

    return rag(query_text)


def transcribe_and_query(audio_path) -> Tuple[str, str, str, str]:
    """Placeholder STT integration.

    In production replace with an actual STT call that returns text.
    """
    # For now simulate an STT transcription
    user_text = "What's the weather like today in London?"
    return rag(user_text)


def _read_css() -> str:
    """Read CSS from the assets folder. Return empty string if missing."""
    css_path = pathlib.Path(__file__).parent / "assets" / "styles.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except Exception:
        return ""


css = _read_css()

# Configure basic logging for the application. Logs go to stderr by default.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### MODEL A")
            out_a = gr.Textbox(label=None, placeholder="Response A...",
                               interactive=False, container=False)
            _ = gr.Button("🔊 Play Audio", size="sm")

        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### MODEL B")
            out_b = gr.Textbox(label=None, placeholder="Response B...",
                               interactive=False, container=False)
            _ = gr.Button("🔊 Play Audio", size="sm")

        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### MODEL C")
            out_c = gr.Textbox(label=None, placeholder="Response C...",
                               interactive=False, container=False)
            _ = gr.Button("🔊 Play Audio", size="sm")

    gr.HTML("<div style='height: 150px;'></div>")

    with gr.Row(elem_id="bottom-bar"):
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                show_label=False,
                placeholder="What's the weather like today in London?",
                container=False,
                interactive=True,
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Send", variant="primary", size="lg")
        with gr.Column(scale=2):
            voice_btn = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record",
                show_label=False,
            )

    # Bind both the button click and pressing Enter in the textbox
    user_input.submit(process_query, inputs=[user_input],
                      outputs=[out_a, out_b, out_c])
    submit_btn.click(process_query, inputs=[user_input],
                     outputs=[user_input, out_a, out_b, out_c])

    voice_btn.stop_recording(transcribe_and_query,
                             inputs=[voice_btn],
                             outputs=[out_a, out_b, out_c])


if __name__ == "__main__":
    demo.launch(css=css)