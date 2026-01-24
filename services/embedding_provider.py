"""Embedding provider wrapper for Cohere.

Provides a small, robust adapter that returns a single vector for a
given text. On failure it returns an empty list so callers can handle
the absence of embeddings.
"""

from typing import List
from config import settings
import cohere
import logging


logger = logging.getLogger(__name__)


class CohereEmbeddingProvider:
    """Adapter around Cohere embeddings API."""

    def __init__(self) -> None:
        self.client = cohere.ClientV2(settings.COHERE_API_KEY)

    def embed(self, text: str) -> List[float]:
        """Return embedding vector for `text`.

        Returns an empty list on error so the calling pipeline can skip
        vector-based search when embedding is unavailable.
        """
        try:
            resp = self.client.embed(
                texts=[text],
                model="embed-v4.0",
                input_type="search_query",
                embedding_types=["float"],
            )
            return resp.embeddings.float_[0]
        except Exception as exc:  # minimal, graceful handling
            logger.exception("Cohere embedding error")
            return []
