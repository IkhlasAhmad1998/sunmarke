# Sunmarke RAG Gradio App

This repository contains a small RAG (Retrieval-Augmented Generation) demo wired to a Gradio UI. The code has been refactored into modular services and a pipeline to make it easy to add/remove providers.

Setup

1. Create a virtual environment and activate it.

Windows (cmd):
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Populate `.env` with your provider credentials (a sample `.env` is included).

Run

```bat
python app.py
```

Files of interest

- `app.py`: Gradio UI wired to the `RAGPipeline`.
- `config.py`: Central settings loader.
- `services/`: Provider wrappers for embedding, search, and models.
- `rag_pipeline.py`: Orchestration logic for embedding -> search -> generation.

Extending

To add a new model provider implement `ModelProvider` in `services/model_providers.py` and instantiate it inside `rag_pipeline.create_default_pipeline` or pass the client when creating the pipeline.
# sunmarke
AI Engineer Technical Assessment Project
