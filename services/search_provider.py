"""Search provider using Weaviate hybrid search.

This module keeps a small surface area: connect at import time and
expose `hybrid_search` which returns a list-like result. On errors it
returns an empty list so the pipeline can continue.
"""

from typing import List, Optional
import weaviate
from weaviate.classes.query import HybridFusion
from config import settings
import logging


logger = logging.getLogger(__name__)


try:
    weav_client = weaviate.WeaviateAsyncClient(
        cluster_url=settings.SUNMARKE_WEAVIATE_URL,
        auth_credentials=settings.SUNMARKE_WEAVIATE_API_KEY,
    )
    collection = weav_client.collections.get(settings.SUNMARKE_COLLECTION)
except Exception as exc:  # keep connection errors graceful
    logger.exception("Weaviate connection error")
    weav_client = None
    collection = None


def hybrid_search(
    query_text: str, query_vector=None, alpha: float = 0.5, limit: int = 2
) -> List:
    """Perform a hybrid search; return empty list on failure."""
    if collection is None:
        return []

    try:
        response = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            limit=limit,
        )
        return response.objects
    except Exception as exc:  # minimal, graceful handling
        logger.exception("Hybrid search error")
        return []
