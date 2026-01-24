from typing import Tuple
from config import settings
from services.embedding_provider import CohereEmbeddingProvider
from services.search_provider import hybrid_search
from services.model_providers import call_deepseek, call_kimi, call_gemini




def rag(query):
    embedding = CohereEmbeddingProvider()
    query_embedding = embedding.embed(query)

    # 2) hybrid search
    relevant_docs = hybrid_search(query, query_embedding)

    # 3) build context
    context = "\n\n".join(str(item.properties) for item in relevant_docs)

    # 4) call models
    deepseek_response = call_deepseek(query, context)
    kimi_response = call_kimi(query, context)
    gemini_response = call_gemini(query, context)

    # Return input plus three responses so Gradio outputs match: [input, out_a, out_b, out_c]
    return query, deepseek_response, kimi_response, gemini_response




