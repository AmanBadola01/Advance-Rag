"""
Hybrid (BM25 + Dense Vector) Retriever
Combines Dense Semantic Vector Search (ChromaDB) and Sparse Lexical Search (BM25)
using Reciprocal Rank Fusion (RRF k=60) with configurable weights.
"""

from typing import List, Dict, Any, Optional
from app.models.schemas import ProcessedChunk, RetrievedChunk, RetrievalFilter
from .basic_vector import BasicVectorRetriever
from .bm25 import BM25Retriever


class HybridRetriever:
    """
    Hybrid Retrieval System:
    Combines dense semantic vector search (ChromaDB) with sparse lexical search (BM25).
    Merges candidate lists using Reciprocal Rank Fusion (RRF) with configurable weights.
    """

    # Initializes the hybrid retriever with dense vector, BM25 sparse index, weights, and RRF k factor.
    def __init__(
        self,
        vector_retriever: BasicVectorRetriever,
        bm25_retriever: Optional[BM25Retriever] = None,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    # Synchronously adds and indexes a batch of chunks into both ChromaDB vector store and BM25 index.
    def add_chunks(self, chunks: List[ProcessedChunk]) -> Dict[str, int]:
        if not chunks:
            return {"vector_added": 0, "bm25_added": 0}

        vector_count = self.vector_retriever.add_chunks(chunks)
        bm25_count = self.bm25_retriever.add_chunks(chunks)

        return {"vector_added": vector_count, "bm25_added": bm25_count}

    # Fuses dense and BM25 ranked result lists using weighted Reciprocal Rank Fusion (k=60).
    def compute_rrf(
        self,
        dense_results: List[RetrievedChunk],
        bm25_results: List[RetrievedChunk],
        semantic_weight: float,
        bm25_weight: float,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion (RRF) Formula:
        RRF_Score(d) = (w_dense / (k + rank_dense)) + (w_bm25 / (k + rank_bm25))
        Where k=60 is the standard smoothing constant.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}
        dense_score_map: Dict[str, float] = {}
        bm25_score_map: Dict[str, float] = {}
        dense_rank_map: Dict[str, int] = {}
        bm25_rank_map: Dict[str, int] = {}

        # 1. Process Dense Vector Candidates
        for rank, chunk in enumerate(dense_results, start=1):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            dense_score_map[chunk_id] = chunk.score
            dense_rank_map[chunk_id] = rank

            score_contribution = semantic_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_contribution

        # 2. Process BM25 Keyword Candidates
        for rank, chunk in enumerate(bm25_results, start=1):
            chunk_id = chunk.chunk_id
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
            bm25_score_map[chunk_id] = chunk.bm25_score or chunk.score
            bm25_rank_map[chunk_id] = rank

            score_contribution = bm25_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_contribution

        # 3. Sort candidates descending by fused RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        fused_chunks: List[RetrievedChunk] = []
        for rank_idx, chunk_id in enumerate(sorted_chunk_ids[:top_k], start=1):
            base_chunk = chunk_map[chunk_id]
            fused_score = round(rrf_scores[chunk_id], 6)

            # Metadata enrichment for Chunk Inspector UI
            inspector_meta = {
                **base_chunk.metadata,
                "dense_rank": dense_rank_map.get(chunk_id),
                "bm25_rank": bm25_rank_map.get(chunk_id),
                "dense_score": dense_score_map.get(chunk_id),
                "bm25_score": bm25_score_map.get(chunk_id),
                "rrf_score": fused_score,
                "retrieval_mode": "hybrid_rrf",
            }

            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=base_chunk.text,
                    score=fused_score,
                    rank=rank_idx,
                    strategy="hybrid",
                    source=base_chunk.source,
                    page=base_chunk.page,
                    section=base_chunk.section,
                    parent_id=base_chunk.parent_id,
                    bm25_score=bm25_score_map.get(chunk_id),
                    metadata=inspector_meta,
                )
            )

        return fused_chunks

    # Executes dense vector and BM25 lexical searches in parallel and merges candidates via RRF.
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
        semantic_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []

        w_dense = semantic_weight if semantic_weight is not None else self.semantic_weight
        w_bm25 = bm25_weight if bm25_weight is not None else self.bm25_weight

        # Retrieve a broader candidate pool from both retrievers (top_k * 2) to ensure healthy overlap
        candidate_pool_size = max(top_k * 2, 10)

        # 1. Dense Semantic Search
        dense_candidates = self.vector_retriever.retrieve(
            query=query,
            top_k=candidate_pool_size,
            filters=filters,
        )

        # 2. Sparse Lexical Search
        bm25_candidates = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_pool_size,
            filters=filters,
        )

        # 3. Fuse results with RRF
        return self.compute_rrf(
            dense_results=dense_candidates,
            bm25_results=bm25_candidates,
            semantic_weight=w_dense,
            bm25_weight=w_bm25,
            top_k=top_k,
        )

    # Wraps the hybrid search pipeline into a native LangChain LCEL-compatible runnable retriever.
    def as_langchain_retriever(
        self,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
        semantic_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ):
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document

        parent_self = self

        class _LangChainHybridRetriever(BaseRetriever):
            """LangChain LCEL adapter for HybridRetriever."""

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> List[Document]:
                chunks = parent_self.retrieve(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    semantic_weight=semantic_weight,
                    bm25_weight=bm25_weight,
                )
                return [
                    Document(
                        page_content=c.text,
                        metadata={
                            **c.metadata,
                            "chunk_id": c.chunk_id,
                            "score": c.score,
                            "rank": c.rank,
                            "bm25_score": c.bm25_score,
                        },
                    )
                    for c in chunks
                ]

        return _LangChainHybridRetriever()

    # Cascades document deletion across both ChromaDB and BM25 indices.
    def delete_by_source(self, source: str) -> None:
        if source:
            self.vector_retriever.delete_by_source(source)
            self.bm25_retriever.delete_by_source(source)

    # Returns chunk count statistics across both dense and sparse storage engines.
    def count(self) -> Dict[str, int]:
        return {
            "vector_chunks": self.vector_retriever.count(),
            "bm25_chunks": self.bm25_retriever.count(),
        }

    # Clears all indexed chunks from both vector and BM25 retrievers.
    def clear(self) -> None:
        self.vector_retriever.clear()
        self.bm25_retriever.clear()

