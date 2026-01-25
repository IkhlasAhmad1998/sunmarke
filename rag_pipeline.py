"""Retrieval-Augmented Generation (RAG) orchestration.

This module is responsible for constructing the retrieval context
for a user query and streaming model responses in parallel for the
three configured providers. It yields incremental updates used by
the UI to display streaming assistant responses.
"""

import asyncio
from typing import AsyncGenerator, List, Dict, Tuple
from services.embedding_provider import CohereEmbeddingProvider
from services.search_provider import hybrid_search
from services.model_providers import call_deepseek, call_kimi, call_gemini
import logging

logger = logging.getLogger(__name__)


async def get_context(query: str) -> str:
    """Internal helper: Embeds query and performs hybrid search."""
    try:
        # 1. Generate Embedding
        embedder = CohereEmbeddingProvider()
        query_vector = embedder.embed(query)

        # 2. Hybrid Search
        relevant_docs = await hybrid_search(query, query_vector)

        # 3. Format Context
        context = "\n\n".join(str(item.properties) for item in relevant_docs)
        return context
    except Exception:
        logger.exception("Context retrieval failed")
        return ""


async def rag_stream(
    query: str,
    hist_a: List[Dict],
    hist_b: List[Dict],
    hist_c: List[Dict]
) -> AsyncGenerator[Tuple[List[Dict], List[Dict], List[Dict]], None]:
    """
    Orchestrates parallel streaming from three models.
    Yields updated history lists for [Model A, Model B, Model C].
    """

    # Step 1: Get Context (Shared for all models)
    context = await get_context(query)

    # Step 2: Prepare History
    # Add the user query to all histories immediately
    hist_a.append({"role": "user", "content": query})
    hist_b.append({"role": "user", "content": query})
    hist_c.append({"role": "user", "content": query})

    # Add placeholder assistant messages that we will update during streaming
    hist_a.append({"role": "assistant", "content": ""})
    hist_b.append({"role": "assistant", "content": ""})
    hist_c.append({"role": "assistant", "content": ""})

    # Step 3: Initialize Async Generators for each model
    # Note: We pass the history EXCLUDING the latest placeholder we just added
    gen_a = call_deepseek(query, context, hist_a[:-2])
    gen_b = call_kimi(query, context, hist_b[:-2])
    gen_c = call_gemini(query, context, hist_c[:-2])

    # Step 4: Parallel Consumption Loop
    # We use a set of tasks to monitor which generator has a new token
    tasks = {
        asyncio.create_task(gen_a.__anext__()): 'a',
        asyncio.create_task(gen_b.__anext__()): 'b',
        asyncio.create_task(gen_c.__anext__()): 'c'
    }

    while tasks:
        done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            label = tasks.pop(task)
            try:
                content = task.result()

                # Update the content in the respective history
                if label == 'a':
                    hist_a[-1]["content"] = content
                    # Re-schedule the next iteration for this generator
                    tasks[asyncio.create_task(gen_a.__anext__())] = 'a'
                elif label == 'b':
                    hist_b[-1]["content"] = content
                    tasks[asyncio.create_task(gen_b.__anext__())] = 'b'
                elif label == 'c':
                    hist_c[-1]["content"] = content
                    tasks[asyncio.create_task(gen_c.__anext__())] = 'c'

                # Yield the current state of all histories to update UI bubbles
                yield hist_a, hist_b, hist_c

            except StopAsyncIteration:
                # Generator finished normally
                pass
            except Exception:
                logger.exception(f"Error in generator {label}")
                # Set error message in history if it failed
                msg = "Model error occurred."
                if label == 'a':
                    hist_a[-1]["content"] = msg
                elif label == 'b':
                    hist_b[-1]["content"] = msg
                elif label == 'c':
                    hist_c[-1]["content"] = msg
                yield hist_a, hist_b, hist_c
