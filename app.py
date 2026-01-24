import gradio as gr
from rag_pipeline import rag
from config import settings


def process_query(query_text: str):
    if not query_text or query_text.strip() == "":
        return "", "", "", ""

    return rag(query_text)


def transcribe_and_query(audio):
    # Placeholder STT: in production integrate a real STT provider
    user_text = "What's the weather like today in London?"
    return rag(user_text)


# UI styling
custom_css = """
.model-column { 
    border-right: 1px solid #e0e0e0; 
    padding-right: 20px; 
    min-height: 400px; 
}
.model-column:last-child { 
    border-right: none; 
}
.response-bubble { 
    border-radius: 15px; 
    padding: 15px; 
    margin-bottom: 10px; 
}
#bottom-bar { 
    position: fixed; 
    bottom: 0; 
    left: 0; 
    width: 100%; 
    background: white; 
    padding: 20px; 
    z-index: 100; 
    border-top: 1px solid #ddd; 
}
"""


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### MODEL A")
            out_a = gr.Textbox(label=None, placeholder="Response A...", interactive=False, container=False)
            btn_a = gr.Button("🔊 Play Audio", size="sm")

        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### MODEL B")
            out_b = gr.Textbox(label=None, placeholder="Response B...", interactive=False, container=False)
            btn_b = gr.Button("🔊 Play Audio", size="sm")

        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### MODEL C")
            out_c = gr.Textbox(label=None, placeholder="Response C...", interactive=False, container=False)
            btn_c = gr.Button("🔊 Play Audio", size="sm")

    gr.HTML("<div style='height: 150px;'></div>")

    with gr.Row(elem_id="bottom-bar"):
        with gr.Column(scale=8):
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

    submit_btn.click(process_query, inputs=[user_input], outputs=[user_input, out_a, out_b, out_c])

    voice_btn.stop_recording(transcribe_and_query, inputs=[voice_btn], outputs=[user_input, out_a, out_b, out_c])


if __name__ == "__main__":
    demo.launch(css=custom_css)