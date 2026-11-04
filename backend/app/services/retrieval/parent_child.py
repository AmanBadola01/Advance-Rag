"""
Parent-Child Context Expansion Retriever.
Searches focused, fine-grained child chunks (200 tokens) in the vector/hybrid store
and automatically expands and returns the rich, surrounding parent chunk (1000 tokens) to the LLM.
"""

from typing import List, Dict, Any, Optional, Union
from app.models.schemas import ProcessedChunk, RetrievedChunk, RetrievalFilter, ParentChildResult
from app.services.chunking.parent_child import ParentChildChunker
from .basic_vector import BasicVectorRetriever
from .hybrid import HybridRetriever


class ParentChildRetriever:
    """
    Parent-Child Context Expansion Retriever:
    1. Indexes small child chunks into a child retriever (Vector or Hybrid) for sharp search.
    2. Stores large parent chunks in an in-memory document store.
    3. On query, retrieves matching child chunks and resolves their parent IDs.
    4. Deduplicates multiple children belonging to the same parent to optimize context window usage.
    """

    # Initializes the retriever with a child retriever engine, parent chunk store, and chunker.
    def __init__(
        self,
        child_retriever: Union[BasicVectorRetriever, HybridRetriever],
        chunker: Optional[ParentChildChunker] = None,
    ):
        self.child_retriever = child_retriever
        self.chunker = chunker or ParentChildChunker()
        self.parent_store: Dict[str, ProcessedChunk] = {}

    # Stores parent chunks in the docstore and indexes child chunks into the search engine.
    def add_parent_child_result(self, result: ParentChildResult) -> Dict[str, int]:
        if not result:
            return {"parent_chunks_added": 0, "child_chunks_added": 0}

        # Store parents in memory map
        for parent in result.parents:
            self.parent_store[parent.chunk_id] = parent

        # Index child chunks into vector or hybrid store
        child_count = self.child_retriever.add_chunks(result.children)

        return {
            "parent_chunks_added": len(result.parents),
            "child_chunks_added": child_count if isinstance(child_count, int) else sum(child_count.values()),
        }

    # Splits raw text into parent-child hierarchy and indexes both levels automatically.
    def add_document(
        self,
        text: str,
        doc_id: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        result = self.chunker.split_into_parent_child(
            text=text,
            doc_id=doc_id,
            source=source,
            custom_metadata=metadata,
        )
        return self.add_parent_child_result(result)

    # Ingests a batch of documents into the parent-child retrieval index.
    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        total_parents = 0
        total_children = 0
        for doc in documents:
            counts = self.add_document(
                text=doc.get("text", ""),
                doc_id=doc.get("doc_id", "unknown"),
                source=doc.get("source", "unknown"),
                metadata=doc.get("metadata"),
            )
            total_parents += counts["parent_chunks_added"]
            total_children += counts["child_chunks_added"]

        return {"parent_chunks_added": total_parents, "child_chunks_added": total_children}

    # Searches child chunks, resolves their parent IDs, deduplicates parents, and returns rich parent chunks.
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
        return_parent_context: bool = True,
    ) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []

        # Pull a larger candidate pool of child chunks to ensure enough unique parents after deduplication
        candidate_pool_size = max(top_k * 3, 15)
        child_candidates = self.child_retriever.retrieve(
            query=query,
            top_k=candidate_pool_size,
            filters=filters,
        )

        if not child_candidates:
            return []

        # If user explicitly requests raw child chunks without parent expansion
        if not return_parent_context:
            return child_candidates[:top_k]

        resolved_parents: List[RetrievedChunk] = []
        seen_parent_ids = set()

        for child in child_candidates:
            parent_id = child.parent_id

            # If no parent_id or parent not found in docstore, fall back to child chunk itself
            if not parent_id or parent_id not in self.parent_store:
                if child.chunk_id not in seen_parent_ids:
                    seen_parent_ids.add(child.chunk_id)
                    resolved_parents.append(child)
                continue

            # Deduplication: If this parent was already retrieved by an earlier, higher-scoring child, skip it
            if parent_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_id)
            parent_chunk = self.parent_store[parent_id]

            # Construct enriched RetrievedChunk carrying full parent text and child match traceability
            resolved_parents.append(
                RetrievedChunk(
                    chunk_id=parent_chunk.chunk_id,
                    text=parent_chunk.text,
                    score=child.score,
                    rank=len(resolved_parents) + 1,
                    strategy="parent_child",
                    source=parent_chunk.metadata.source,
                    page=parent_chunk.metadata.page or child.page,
                    section=parent_chunk.metadata.section,
                    parent_id=None,
                    bm25_score=child.bm25_score,
                    metadata={
                        **parent_chunk.metadata.custom_metadata,
                        "matched_child_id": child.chunk_id,
                        "matched_child_rank": child.rank,
                        "matched_child_score": child.score,
                        "matched_child_snippet": child.text[:120] + ("..." if len(child.text) > 120 else ""),
                        "parent_token_count": parent_chunk.metadata.token_count,
                        "child_token_count": child.metadata.get("token_count"),
                        "retrieval_mode": "parent_context_expansion",
                    },
                )
            )

            if len(resolved_parents) >= top_k:
                break

        return resolved_parents

    # Exposes the parent-child retriever as a native LangChain LCEL-compatible runnable retriever.
    def as_langchain_retriever(
        self,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
        return_parent_context: bool = True,
    ):
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document

        parent_self = self

        class _LangChainParentChildRetriever(BaseRetriever):
            """LangChain LCEL adapter for ParentChildRetriever."""

            # Executes parent-child retrieval and formats resolved parent chunks into LangChain Documents.
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
                    return_parent_context=return_parent_context,
                )
                return [
                    Document(
                        page_content=c.text,
                        metadata={
                            **c.metadata,
                            "chunk_id": c.chunk_id,
                            "score": c.score,
                            "rank": c.rank,
                            "strategy": c.strategy,
                            "source": c.source,
                            "matched_child_id": c.metadata.get("matched_child_id"),
                        },
                    )
                    for c in chunks
                ]

        return _LangChainParentChildRetriever()

    # Removes all parent chunks and corresponding child chunks associated with a specific document source.
    def delete_by_source(self, source: str) -> None:
        if not source:
            return

        # Remove matching parents from in-memory docstore
        to_delete = [pid for pid, p in self.parent_store.items() if p.metadata.source == source]
        for pid in to_delete:
            del self.parent_store[pid]

        # Cascade deletion down to child retriever
        self.child_retriever.delete_by_source(source)

    # Returns the count of indexed parent chunks and underlying child chunks.
    def count(self) -> Dict[str, Any]:
        return {
            "parent_chunks": len(self.parent_store),
            "child_chunks": self.child_retriever.count(),
        }

    # Clears all parent and child chunks from storage.
    def clear(self) -> None:
        self.parent_store.clear()
        self.child_retriever.clear()

