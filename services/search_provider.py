from typing import List, Optional
import weaviate
from weaviate.classes.query import HybridFusion
from config import settings


weav_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=settings.SUNMARKE_WEAVIATE_URL,
    auth_credentials=settings.SUNMARKE_WEAVIATE_API_KEY
)

collection = weav_client.collections.get(settings.SUNMARKE_COLLECTION)

def hybrid_search(query_text: str, query_vector=None, alpha: float = 0.5, limit: int = 2):

    response = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=alpha,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        limit=limit,
    )
    return response.objects
