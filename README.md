# Sunmarke RAG Gradio App
# Sunmarke RAG Gradio App

Small Retrieval-Augmented Generation (RAG) demo with a Gradio UI.
The project is split into a UI layer (`app.py`), a lightweight
RAG orchestration (`rag_pipeline.py`), and provider adapters under
`services/` (embeddings, search, models, and voice).

**Architecture**
- `app.py`: Gradio-based web UI and event wiring. Handles user input,
	displays three parallel model responses, and integrates microphone
	recording.
- `rag_pipeline.py`: Orchestrates embedding -> hybrid search -> model
	generation. Streams partial responses from multiple providers.
- `services/embedding_provider.py`: Embedding adapter (Cohere).
- `services/search_provider.py`: Async Weaviate hybrid search wrapper.
- `services/model_providers.py`: Async streaming adapters for each
	model provider (OpenRouter/Deepseek, Groq/Kimi, Google Gemini).
- `services/voice_service.py`: Audio transcription using Deepgram.
- `services/prompts.py`: Centralized system prompt and policy for
	responses.
- `data`: Contains notebooks for webscraping, chunking, embedding and ingestion.

**Installation**
1. Create and activate a virtual environment.

Windows (cmd):
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Provide credentials via a `.env` file in the project root. The
	 repository expects the following environment variables (based on
	 `config.py`):

- `SUNMARKE_WEAVIATE_API_KEY`
- `SUNMARKE_WEAVIATE_URL`
- `SUNMARKE_COLLECTION`
- `COHERE_API_KEY`
- `GEMINI_API_KEY`
- `OPEN_ROUTER_URL`
- `OPEN_ROUTER_API_KEY`
- `GROQ_API_KEY`
- `GROQ_BASE_URL`
- `DEEPGRAM_API_KEY`

Keep secrets out of source control and use a secure vault for
production deployment.

**Run (development)**
```bat
python app.py
```

The Gradio UI will launch and expose a local URL for interaction.

**Extending & Development notes**
- Add new embedding or model providers under `services/` and expose a
	small async helper that streams tokens (see `model_providers.py`).
- Keep business logic in `rag_pipeline.py` and adapters in
	`services/` to maintain separation of concerns.
- The repository uses async clients to avoid blocking the UI; unit
	tests and additional error handling are recommended before
	production deployment.

**Files of interest**
- [app.py](app.py): UI and event wiring.
- [config.py](config.py): Environment-based settings.
- [rag_pipeline.py](rag_pipeline.py): RAG orchestration logic.
- [services/](services/): Provider adapters and helpers.
- [data/](data/): Provider webscraping, chunking, embedding and ingestion notebooks.

---
AI Engineer Technical Assessment Project
