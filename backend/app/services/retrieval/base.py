"""
Base Retrieval Infrastructure
Provides shared embedding wrapper, filter conversion utilities, and base interface.
"""

from typing import List, Dict, Any, Optional
from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from app.models.schemas import RetrievalFilter, RetrievedChunk


class LocalMiniLMEmbedding(EmbeddingFunction):
    """
    Local SentenceTransformer embedding wrapper for ChromaDB.
    Uses all-MiniLM-L6-v2 (384 dimensions) completely offline and free.
    """

    # Initializes the SentenceTransformer embedding model from local cache.
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    # Lazy-loads the PyTorch model on first invocation to optimize startup time.
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        return self._model

    # Encodes input texts into dense embedding float vectors.
    def __call__(self, input: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    # Returns standard identifier for ChromaDB embedding function registry.
    @staticmethod
    def name() -> str:
        return "local_minilm_l6_v2"

    # Returns configuration parameters for collection serialization.
    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    # Deserializes and restores the embedding instance from collection configuration.
    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "LocalMiniLMEmbedding":
        return LocalMiniLMEmbedding(model_name=config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))


# Builds a ChromaDB boolean filter dictionary from a RetrievalFilter schema.
def build_chroma_filter(filter_params: Optional[RetrievalFilter]) -> Optional[Dict[str, Any]]:
    """
    Converts RetrievalFilter into ChromaDB 'where' query syntax:
    Supports source, page range, section, and strategy with $and combinations.
    """
    if not filter_params:
        return None

    conditions: List[Dict[str, Any]] = []

    if filter_params.source:
        conditions.append({"source": {"$eq": filter_params.source}})

    if filter_params.strategy:
        conditions.append({"strategy": {"$eq": filter_params.strategy}})

    if filter_params.section:
        conditions.append({"section": {"$eq": filter_params.section}})

    if filter_params.page is not None:
        conditions.append({"page": {"$eq": filter_params.page}})
    else:
        if filter_params.page_min is not None and filter_params.page_max is not None:
            conditions.append({"page": {"$gte": filter_params.page_min}})
            conditions.append({"page": {"$lte": filter_params.page_max}})
        elif filter_params.page_min is not None:
            conditions.append({"page": {"$gte": filter_params.page_min}})
        elif filter_params.page_max is not None:
            conditions.append({"page": {"$lte": filter_params.page_max}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
