from typing import List
from config import settings
import cohere


class CohereEmbeddingProvider:
    def __init__(self):
        self.client = cohere.ClientV2(settings.COHERE_API_KEY)

    def embed(self, texts: List[str]) -> List[List[float]]:

        resp = self.client.embed(
            texts=[texts],
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
        )
        return resp.embeddings.float_[0]
