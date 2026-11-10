"""
Cross-Encoder Re-Ranker and Two-Stage Hybrid Re-Rank Retriever.
Implements Stage 2 deep cross-attention re-ranking using ms-marco-MiniLM-L-6-v2.
Takes Stage 1 candidates from Hybrid Search and re-scores them for maximum precision.
"""

import math
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

from app.models.schemas import ProcessedChunk, RetrievedChunk, RetrievalFilter
from .hybrid import HybridRetriever


# Computes numerically stable sigmoid to map unbounded logit scores into a clean 0.0 to 1.0 range.
def sigmoid_score(x: float) -> float:
    clamped_x = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + math.exp(-clamped_x))


class CrossEncoderReranker:
    """
    Standalone Cross-Encoder Re-Ranker:
    Loads a local transformer cross-encoder (ms-marco-MiniLM-L-6-v2) to evaluate
    query-chunk pairs simultaneously using cross-attention.
    """

    # Initializes the CrossEncoder model with local cache fallback for offline execution.
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    # Lazy-loads the CrossEncoder PyTorch model on first invocation to optimize startup.
    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, local_files_only=os.getenv("RAG_MODEL_LOCAL_ONLY", "false").lower() == "true")
        return self._model

    # Scores candidate chunks against the user query using cross-attention and returns sorted top-k results.
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if not candidates or not query or not query.strip():
            return []

        # If candidates are fewer than or equal to top_k, we still re-score them to ensure optimal order
        model = self._get_model()

        # Construct (query, passage) text pairs for the transformer
        query_chunk_pairs = [[query, chunk.text] for chunk in candidates]

        # Compute raw logits via cross-attention
        raw_scores = model.predict(query_chunk_pairs)

        # Pair each candidate with its new cross-encoder score and original rank
        scored_candidates = []
        for original_rank, (chunk, raw_score) in enumerate(zip(candidates, raw_scores), start=1):
            normalized_score = round(sigmoid_score(raw_score), 4)
            scored_candidates.append({
                "chunk": chunk,
                "original_rank": original_rank,
                "raw_score": float(raw_score),
                "normalized_score": normalized_score,
            })

        # Sort descending by re-rank score
        scored_candidates.sort(key=lambda item: item["normalized_score"], reverse=True)

        reranked_results: List[RetrievedChunk] = []
        for new_rank, item in enumerate(scored_candidates[:top_k], start=1):
            base_chunk = item["chunk"]
            enriched_metadata = {
                **base_chunk.metadata,
                "pre_rerank_rank": item["original_rank"],
                "rerank_score": item["normalized_score"],
                "raw_cross_score": round(item["raw_score"], 4),
                "retrieval_mode": "hybrid_rerank",
            }

            reranked_results.append(
                RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    text=base_chunk.text,
                    score=item["normalized_score"],
                    rank=new_rank,
                    strategy="hybrid_rerank",
                    source=base_chunk.source,
                    page=base_chunk.page,
                    section=base_chunk.section,
                    parent_id=base_chunk.parent_id,
                    bm25_score=base_chunk.bm25_score,
                    metadata=enriched_metadata,
                )
            )

        return reranked_results


class HybridRerankRetriever:
    """
    Two-Stage Hybrid + Re-Ranking Retriever:
    Stage 1: Broad candidate collection via HybridRetriever (Vector + BM25 via RRF).
    Stage 2: Deep cross-encoder re-ranking via CrossEncoderReranker down to top_k.
    """

    # Initializes the two-stage retriever with hybrid candidate retriever and cross-encoder re-ranker.
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        candidate_pool_size: int = 20,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_pool_size = candidate_pool_size

    # Indexes a batch of document chunks synchronously into both vector store and BM25 index.
    def add_chunks(self, chunks: List[ProcessedChunk]) -> Dict[str, int]:
        return self.hybrid_retriever.add_chunks(chunks)

    # Executes two-stage retrieval: broad hybrid search followed by deep cross-encoder re-ranking.
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

        # Stage 1: Retrieve broad candidate pool (top 20)
        pool_size = max(self.candidate_pool_size, top_k * 2)
        candidates = self.hybrid_retriever.retrieve(
            query=query,
            top_k=pool_size,
            filters=filters,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

        if not candidates:
            return []

        # Stage 2: Deep cross-encoder re-ranking down to final top_k
        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    # Exposes the two-stage hybrid re-ranker as a native LangChain LCEL-compatible runnable retriever.
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

        class _LangChainHybridRerankRetriever(BaseRetriever):
            """LangChain LCEL adapter for HybridRerankRetriever."""

            # Executes two-stage retrieval and converts output into LangChain Document instances.
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
                            "pre_rerank_rank": c.metadata.get("pre_rerank_rank"),
                            "raw_cross_score": c.metadata.get("raw_cross_score"),
                        },
                    )
                    for c in chunks
                ]

        return _LangChainHybridRerankRetriever()

    # Removes all document chunks belonging to a specific source across both vector and BM25 stores.
    def delete_by_source(self, source: str) -> None:
        self.hybrid_retriever.delete_by_source(source)

    # Returns the chunk count statistics across the underlying retrieval stores.
    def count(self) -> Dict[str, int]:
        return self.hybrid_retriever.count()

    # Clears all indexed document chunks from both vector and BM25 stores.
    def clear(self) -> None:
        self.hybrid_retriever.clear()

