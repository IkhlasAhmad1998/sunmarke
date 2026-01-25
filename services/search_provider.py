import logging
from typing import List, Optional
import weaviate
from weaviate.classes.query import HybridFusion
from config import settings

logger = logging.getLogger(__name__)

# Persistent client placeholder
_async_client: Optional[weaviate.WeaviateAsyncClient] = None


async def get_client() -> weaviate.WeaviateAsyncClient:
    """Singleton-style helper to get or initialize the async Weaviate client."""
    global _async_client
    # Removed 'await' from is_connected()
    if _async_client is None or not _async_client.is_connected():
        try:
            _async_client = weaviate.use_async_with_weaviate_cloud(
                cluster_url=settings.SUNMARKE_WEAVIATE_URL,
                auth_credentials=weaviate.auth.AuthApiKey(settings.SUNMARKE_WEAVIATE_API_KEY),
            )
            await _async_client.connect()
            logger.info("Successfully connected to Weaviate Async Client.")
        except Exception:
            logger.exception("Failed to connect to Weaviate")
            _async_client = None
    return _async_client


async def hybrid_search(
    query_text: str,
    query_vector: List[float] = None,
    alpha: float = 0.5,
    limit: int = 3
) -> List:
    """Perform an asynchronous hybrid search.

    Returns a list of objects or an empty list on failure.
    """
    client = await get_client()
    if client is None:
        return []

    try:
        collection = client.collections.get(settings.SUNMARKE_COLLECTION)

        # v4 Async query syntax
        response = await collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            limit=limit,
        )
        return response.objects
    except Exception:
        logger.exception("Hybrid search execution error")
        return []


async def close_search_client():
    """Cleanup function to be called when the app shuts down."""
    global _async_client
    if _async_client:
        await _async_client.close()
        _async_client = None
