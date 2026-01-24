from typing import Tuple
from config import settings
from services.embedding_provider import CohereEmbeddingProvider
from services.search_provider import hybrid_search
from services.model_providers import call_deepseek, call_kimi, call_gemini


def rag(query: str) -> Tuple[str, str, str, str]:
    """Run a simple RAG flow: embed, hybrid search, then call models.

    The function is defensive: individual components may fail and
    return graceful defaults so the UI stays responsive.
    """
    # 1) embedding (may fail and return empty list)
    embedding = CohereEmbeddingProvider()
    try:
        query_embedding = embedding.embed(query)
    except Exception:
        query_embedding = []

    # 2) hybrid search (returns empty list on failure)
    relevant_docs = hybrid_search(query, query_embedding)

    # 3) build context
    try:
        context = "\n\n".join(str(item.properties) for item in relevant_docs)
    except Exception:
        context = ""

    # 4) call models (each returns an unavailable message on failure)
    deepseek_response = call_deepseek(query, context)
    kimi_response = call_kimi(query, context)
    gemini_response = call_gemini(query, context)

    # Return input plus three responses so Gradio outputs match: [input, out_a, out_b, out_c]
    return deepseek_response, kimi_response, gemini_response
