from config import settings
from typing import Optional
from openai import OpenAI
from groq import Groq
from google import genai

open_router = OpenAI(
    base_url=settings.OPEN_ROUTER_URL,
    api_key=settings.OPEN_ROUTER_API_KEY,
)

groq_client = Groq(
    api_key=settings.GROQ_API_KEY,
    base_url=settings.GROQ_BASE_URL
)

google_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def call_deepseek(query, context):
    completion = open_router.chat.completions.create(
        extra_body={},
        model="deepseek/deepseek-r1-0528:free",
        messages=[
            {
              "role": "user",
              "content": f"answer the following query: {query} from the provided context: CONTEXT: {context}"
            }
        ]
    )

    return completion.choices[0].message.content


def call_kimi(query, context):
    completion = groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[
            {
                "role": "user",
                "content": f"answer the following query: {query} from the provided context: CONTEXT: {context}"
            }
        ]
    )

    return completion.choices[0].message.content


def call_gemini(query, context):
    response = google_client.models.generate_content(
        model="gemini-2.5-flash",   # Gemini model name
        contents=f"answer the following query: {query} from the provided context: CONTEXT: {context}"
    )

    return response.text