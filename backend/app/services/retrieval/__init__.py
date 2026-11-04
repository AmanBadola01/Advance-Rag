"""
Retrieval Suite Package
Implements:
- Basic Vector Retriever (ChromaDB + all-MiniLM-L6-v2)
- BM25 Sparse Keyword Retriever (rank_bm25)
- Hybrid Ensemble Retriever with RRF (BM25 + Vector)
"""

from .base import LocalMiniLMEmbedding, build_chroma_filter
from .basic_vector import BasicVectorRetriever
from .bm25 import BM25Retriever
from .hybrid import HybridRetriever

__all__ = [
    "LocalMiniLMEmbedding",
    "build_chroma_filter",
    "BasicVectorRetriever",
    "BM25Retriever",
    "HybridRetriever",
]
