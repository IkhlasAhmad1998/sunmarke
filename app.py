import gradio as gr
import asyncio
from rag_pipeline import rag_stream

# --- UI Helpers ---
def lock_input():
    """Disables the input bar while processing."""
    return gr.update(interactive=False, placeholder="Thinking..."), gr.update(interactive=False)

def unlock_input():
    """Re-enables the input bar after response."""
    return gr.update(interactive=True, placeholder="Ask a follow-up question..."), gr.update(interactive=True)

async def chat_wrapper(query, hist_a, hist_b, hist_c):
    if not query or query.strip() == "":
        yield hist_a, hist_b, hist_c
        return

    # rag_stream returns an AsyncGenerator yielding (hist_a, hist_b, hist_c)
    async for updated_a, updated_b, updated_c in rag_stream(query, hist_a, hist_b, hist_c):
        yield updated_a, updated_b, updated_c

# --- Layout and UI ---
with gr.Blocks() as demo:
    # Session State for History
    history_a = gr.State([])
    history_b = gr.State([])
    history_c = gr.State([])

    gr.Markdown("## 🤖 Multi-Model RAG Comparison")

    with gr.Row():
        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### DEEPSEEK (Model A)")
            # Removed 'type="messages"' as it's now default in v6.0
            chat_a = gr.Chatbot(label=None, height=500)
            
        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### KIMI (Model B)")
            chat_b = gr.Chatbot(label=None, height=500)
            
        with gr.Column(elem_classes="model-column"):
            gr.Markdown("### GEMINI (Model C)")
            chat_c = gr.Chatbot(label=None, height=500)

    # Bottom Fixed Input Bar
    with gr.Row(elem_id="bottom-bar"):
        with gr.Column(scale=5):
            user_input = gr.Textbox(
                show_label=False, 
                placeholder="Ask something...", 
                container=False
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Send", variant="primary")

    # --- Event Logic ---
    submit_event = (
        submit_btn.click(lock_input, outputs=[user_input, submit_btn])
        .then(chat_wrapper, 
              inputs=[user_input, chat_a, chat_b, chat_c], 
              outputs=[chat_a, chat_b, chat_c])
        .then(lambda: "", outputs=[user_input])
        .then(unlock_input, outputs=[user_input, submit_btn])
    )
    
    user_input.submit(lock_input, outputs=[user_input, submit_btn]).then(
        chat_wrapper, 
        inputs=[user_input, chat_a, chat_b, chat_c], 
        outputs=[chat_a, chat_b, chat_c]
    ).then(lambda: "", outputs=[user_input]).then(unlock_input, outputs=[user_input, submit_btn])

# --- Custom Styling (Moved to launch() per Gradio 6.0) ---
CSS = """
    #bottom-bar { position: fixed; bottom: 0; left: 0; width: 100%; background: white; padding: 20px; z-index: 1000; border-top: 1px solid #ddd; }
    .model-column { border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; background: #f9f9f9; }
    .gradio-container { padding-bottom: 180px !important; } 
"""

if __name__ == "__main__":
    # CSS is now passed here
    demo.launch(css=CSS)