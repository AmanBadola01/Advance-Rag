"""
BM25 Lexical Keyword Retriever
Implements sparse lexical search using the rank_bm25 library (BM25Okapi).
Finds exact keyword matches, product codes, IDs, numbers, and acronyms.
"""

import math
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from app.models.schemas import ProcessedChunk, RetrievedChunk, RetrievalFilter


# BM25 variant using Lucene non-negative IDF formula to guarantee positive scores on small or medium corpora.
class LuceneBM25(BM25Okapi):
    # Calculates term inverse document frequency with Lucene 1+ smoothing to guarantee positive weights.
    def _calc_idf(self, nd: Dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


# Tokenizes a text string into clean lowercase keyword words for BM25 indexing.
def tokenize_text(text: str) -> List[str]:
    """
    Cleans and tokenizes text for lexical search:
    1. Lowercases all characters.
    2. Extracts alphanumeric tokens and hyphenated terms (e.g. 'e-4021', 'gpt-4o').
    """
    if not text:
        return []
    # Match alphanumeric words and words with internal hyphens/underscores
    tokens = re.findall(r"\b[\w\-]+\b", text.lower())
    return tokens


class BM25Retriever:
    """
    BM25 Sparse Lexical Retriever:
    1. Maintains an in-memory inverted index of document chunks.
    2. Calculates Okapi BM25 term frequency-inverse document frequency scores.
    3. Handles exact keyword, numerical, and identifier lookups.
    4. Supports metadata filtering before rank ordering.
    """

    # Initializes an empty BM25 index and chunk storage list.
    def __init__(self):
        self.chunks: List[ProcessedChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[LuceneBM25] = None

    # Tokenizes and indexes a list of processed text chunks into the BM25 inverted index.
    def add_chunks(self, chunks: List[ProcessedChunk]) -> int:
        if not chunks:
            return 0

        new_chunks = [c for c in chunks if c.text and c.text.strip()]
        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)
        # Tokenize each new chunk for BM25
        for c in new_chunks:
            self.tokenized_corpus.append(tokenize_text(c.text))

        # Re-build the LuceneBM25 index with the full corpus
        self.bm25 = LuceneBM25(self.tokenized_corpus)
        return len(new_chunks)

    # Checks whether a chunk satisfies all active metadata filter conditions.
    def _matches_filters(self, chunk: ProcessedChunk, filters: Optional[RetrievalFilter]) -> bool:
        if not filters:
            return True

        meta = chunk.metadata

        # Source document check
        if filters.source and meta.source != filters.source:
            return False

        # Strategy check
        if filters.strategy and meta.strategy != filters.strategy:
            return False

        # Section check
        if filters.section and meta.section != filters.section:
            return False

        # Page checks
        if filters.page is not None and meta.page != filters.page:
            return False
        if filters.page_min is not None and (meta.page is None or meta.page < filters.page_min):
            return False
        if filters.page_max is not None and (meta.page is None or meta.page > filters.page_max):
            return False

        return True

    # Executes BM25 lexical search and returns top-k keyword matching chunks with raw and normalized scores.
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
    ) -> List[RetrievedChunk]:
        if not query or not query.strip() or not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        # Get raw BM25 scores across all indexed chunks
        raw_scores = self.bm25.get_scores(tokenized_query)

        # Pair each chunk with its score and index
        scored_candidates = []
        for idx, score in enumerate(raw_scores):
            chunk = self.chunks[idx]
            # Apply metadata filters
            if self._matches_filters(chunk, filters):
                scored_candidates.append((score, idx, chunk))

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        if not top_candidates:
            return []

        # Determine highest score for normalization (0.0 to 1.0 scale)
        max_score = top_candidates[0][0] if top_candidates[0][0] > 0 else 1.0

        results: List[RetrievedChunk] = []
        for rank_idx, (score, _, chunk) in enumerate(top_candidates):
            # If the score is 0, it means no query keywords matched at all
            if score <= 0:
                continue

            # Normalize raw BM25 score relative to max score
            normalized_score = round(min(1.0, score / max_score), 4)

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=normalized_score,
                    rank=rank_idx + 1,
                    strategy=chunk.metadata.strategy,
                    source=chunk.metadata.source,
                    page=chunk.metadata.page or 1,
                    section=chunk.metadata.section,
                    parent_id=chunk.metadata.parent_id,
                    bm25_score=round(float(score), 4),
                    metadata={
                        **chunk.metadata.custom_metadata,
                        "raw_bm25_score": round(float(score), 4),
                        "token_count": chunk.metadata.token_count,
                    },
                )
            )

        return results

    # Exposes the BM25 index as a native LangChain LCEL-compatible runnable retriever.
    def as_langchain_retriever(
        self,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
    ):
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document

        parent_self = self

        class _LangChainBM25Retriever(BaseRetriever):
            """LangChain LCEL adapter for BM25Retriever."""

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: Optional[CallbackManagerForRetrieverRun] = None,
            ) -> List[Document]:
                chunks = parent_self.retrieve(query=query, top_k=top_k, filters=filters)
                return [
                    Document(
                        page_content=c.text,
                        metadata={
                            **c.metadata,
                            "chunk_id": c.chunk_id,
                            "bm25_score": c.bm25_score,
                            "score": c.score,
                            "rank": c.rank,
                        },
                    )
                    for c in chunks
                ]

        return _LangChainBM25Retriever()

    # Removes all chunks associated with a specific document source and rebuilds the BM25 index.
    def delete_by_source(self, source: str) -> None:
        if not source:
            return

        remaining_chunks = [c for c in self.chunks if c.metadata.source != source]
        self.clear()
        if remaining_chunks:
            self.add_chunks(remaining_chunks)

    # Returns the total number of chunks currently indexed in the BM25 model.
    def count(self) -> int:
        return len(self.chunks)

    # Clears all indexed chunks and resets the BM25 model.
    def clear(self) -> None:
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

