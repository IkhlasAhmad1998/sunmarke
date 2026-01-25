"""Configuration loader.

Loads environment variables from `.env` and exposes a small
`Settings` holder used across the project. Keep secrets out of
source control and store them in the environment or a protected
.env file during local development.
"""

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env'))


class Settings:
    SUNMARKE_WEAVIATE_API_KEY: str | None = os.getenv('SUNMARKE_WEAVIATE_API_KEY')
    SUNMARKE_WEAVIATE_URL: str | None = os.getenv('SUNMARKE_WEAVIATE_URL')
    SUNMARKE_COLLECTION: str | None = os.getenv('SUNMARKE_COLLECTION')

    COHERE_API_KEY: str | None = os.getenv('COHERE_API_KEY')
    GEMINI_API_KEY: str | None = os.getenv('GEMINI_API_KEY')
    OPEN_ROUTER_URL: str | None = os.getenv('OPEN_ROUTER_URL')
    OPEN_ROUTER_API_KEY: str | None = os.getenv('OPEN_ROUTER_API_KEY')
    GROQ_API_KEY: str | None = os.getenv('GROQ_API_KEY')
    GROQ_BASE_URL: str | None = os.getenv('GROQ_BASE_URL')

    DEEPGRAM_API_KEY: str | None = os.getenv("DEEPGRAM_API_KEY")


settings = Settings()
