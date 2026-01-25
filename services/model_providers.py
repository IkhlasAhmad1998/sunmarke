"""Model provider wrappers.

Contains thin async adapters over the external model provider SDKs.
Each function streams partial outputs and yields progressively
concatenated text for the UI to display while the model generates.
"""

import logging
from typing import AsyncGenerator, List, Dict
from openai import AsyncOpenAI
from groq import AsyncGroq
from google import genai
from config import settings
from services.prompts import system_prompt

logger = logging.getLogger(__name__)

# Initialize Clients
# Note: Using Async versions of all clients to prevent UI blocking
open_router = AsyncOpenAI(
    base_url=settings.OPEN_ROUTER_URL,
    api_key=settings.OPEN_ROUTER_API_KEY,
)

groq_client = AsyncGroq(
    api_key=settings.GROQ_API_KEY,
    base_url=settings.GROQ_BASE_URL,
)

# Gemini's new 2025 SDK uses .aio for async operations
google_client = genai.Client(api_key=settings.GEMINI_API_KEY).aio

_UNAVAILABLE_MSG = "Model currently unavailable, try again later."


def _build_messages(query: str, context: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Helper to construct the message list with history and context."""

    messages = [{"role": "system", "content": system_prompt}]

    # SANITIZE HISTORY: Only keep 'role' and 'content' for the API calls
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    messages.append({"role": "user", "content": f"CONTEXT: {context}\n\nQuery: {query}"})
    return messages


async def call_deepseek(query: str, context: str, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Stream response from DeepSeek via OpenRouter."""
    try:
        messages = _build_messages(query, context, history)
        stream = await open_router.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=messages,
            stream=True,
        )
        full_content = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                yield full_content
    except Exception:
        logger.exception("Deepseek streaming error")
        yield _UNAVAILABLE_MSG


async def call_kimi(query: str, context: str, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Stream response from Kimi via Groq."""
    try:
        messages = _build_messages(query, context, history)
        stream = await groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=messages,
            stream=True,
        )
        full_content = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                yield full_content
    except Exception:
        logger.exception("Kimi streaming error")
        yield _UNAVAILABLE_MSG


async def call_gemini(query: str, context: str, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Stream response from Google Gemini."""
    try:
        # Prompt construction for Gemini
        prompt = (
            f"Context: {context}\n\n"
            f"History: {history}\n\n"
            f"User Query: {query}"
        )

        stream = await google_client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        full_content = ""
        async for chunk in stream:
            if chunk.text:
                full_content += chunk.text
                yield full_content
    except Exception:
        logger.exception("Gemini streaming error")
        yield _UNAVAILABLE_MSG
