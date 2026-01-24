"""Model provider adapters.

Wraps calls to external LLM providers. Each adapter returns a simple
string response; on failure a friendly unavailable message is
returned so the UI can display it.
"""

from config import settings
from typing import Optional
from openai import OpenAI
from groq import AsyncClient
from google import genai
import logging


logger = logging.getLogger(__name__)


open_router = OpenAI(
    base_url=settings.OPEN_ROUTER_URL,
    api_key=settings.OPEN_ROUTER_API_KEY,
)


groq_client = AsyncClient(
    api_key=settings.GROQ_API_KEY,
    base_url=settings.GROQ_BASE_URL,
)


google_client = genai.Client(api_key=settings.GEMINI_API_KEY)


_UNAVAILABLE_MSG = "Model currently unavailable, try again later."


def call_deepseek(query: str, context: str) -> str:
    """Call OpenRouter / Deepseek model and return text or fallback."""
    try:
        completion = open_router.chat.completions.create(
            extra_body={},
            model="deepseek/deepseek-r1-0528:free",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"answer the following query: {query} from the provided "
                        f"context: CONTEXT: {context}"
                    ),
                }
            ],
        )
        return completion.choices[0].message.content
    except Exception as exc:  # keep failures graceful
        logger.exception("Deepseek call error")
        return _UNAVAILABLE_MSG


def call_kimi(query: str, context: str) -> str:
    """Call Groq Kimi model and return text or fallback."""
    try:
        completion = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"answer the following query: {query} from the provided "
                        f"context: CONTEXT: {context}"
                    ),
                }
            ],
        )
        return completion.choices[0].message.content
    except Exception as exc:
        logger.exception("Kimi call error")
        return _UNAVAILABLE_MSG


def call_gemini(query: str, context: str) -> str:
    """Call Google Gemini and return text or fallback."""
    try:
        response = google_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                f"answer the following query: {query} from the provided "
                f"context: CONTEXT: {context}"
            ),
        )
        return response.text
    except Exception as exc:
        logger.exception("Gemini call error")
        return _UNAVAILABLE_MSG