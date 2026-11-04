"""
Basic Vector Retriever
Implements standard dense vector similarity search using ChromaDB and all-MiniLM-L6-v2 embeddings.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection

from app.models.schemas import ProcessedChunk, RetrievedChunk, RetrievalFilter
from .base import LocalMiniLMEmbedding, build_chroma_filter


class BasicVectorRetriever:
    """
    Basic Vector Retriever:
    1. Embeds text chunks using dense representations (all-MiniLM-L6-v2, 384 dimensions).
    2. Performs cosine similarity search in ChromaDB.
    3. Supports strict metadata filtering (source, section, page, strategy).
    4. Provides LangChain LCEL-compatible retriever interface.
    """

    # Initializes the ChromaDB client, local embedding function, and target vector collection.
    def __init__(
        self,
        collection_name: str = "rag_chunks",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        client: Optional[chromadb.ClientAPI] = None,
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function or LocalMiniLMEmbedding()

        # Connect to persistent storage directory or transient in-memory client
        if client:
            self.client = client
        elif persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Initialize or get existing ChromaDB collection
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    # Embeds and inserts a batch of processed chunks into the ChromaDB collection with rich metadata.
    def add_chunks(self, chunks: List[ProcessedChunk]) -> int:
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in chunks:
            ids.append(c.chunk_id)
            documents.append(c.text)

            meta: Dict[str, Any] = {
                "chunk_id": c.chunk_id,
                "strategy": c.metadata.strategy,
                "source": c.metadata.source,
                "page": c.metadata.page or 1,
                "token_count": c.metadata.token_count,
                "char_count": c.metadata.char_count,
            }
            if c.metadata.section:
                meta["section"] = c.metadata.section
            if c.metadata.parent_id:
                meta["parent_id"] = c.metadata.parent_id
            if c.metadata.file_type:
                meta["file_type"] = c.metadata.file_type

            metadatas.append(meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    # Executes dense cosine similarity search in ChromaDB and returns ranked results with similarity scores.
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
    ) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []

        where_clause = build_chroma_filter(filters)

        # Query ChromaDB dense index
        query_kwargs: Dict[str, Any] = {
            "query_texts": [query.strip()],
            "n_results": top_k,
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self.collection.query(**query_kwargs)

        retrieved: List[RetrievedChunk] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for rank_idx, chunk_id in enumerate(ids):
            text = docs[rank_idx] if rank_idx < len(docs) else ""
            meta = metas[rank_idx] if rank_idx < len(metas) else {}
            distance = distances[rank_idx] if rank_idx < len(distances) else 0.0

            # Convert cosine distance to cosine similarity score: similarity = 1 - distance
            similarity_score = max(0.0, min(1.0, 1.0 - distance))

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=round(similarity_score, 4),
                    rank=rank_idx + 1,
                    strategy=str(meta.get("strategy", "vector")),
                    source=str(meta.get("source", "unknown")),
                    page=int(meta.get("page", 1)),
                    section=meta.get("section"),
                    parent_id=meta.get("parent_id"),
                    metadata=meta,
                )
            )

        return retrieved

    # Exposes the vector collection as a native LangChain LCEL-compatible runnable retriever.
    def as_langchain_retriever(
        self,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
    ):
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document

        parent_self = self

        class _LangChainVectorRetriever(BaseRetriever):
            """LangChain LCEL adapter for BasicVectorRetriever."""
            
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
                            "score": c.score,
                            "rank": c.rank,
                        },
                    )
                    for c in chunks
                ]

        return _LangChainVectorRetriever()

    # Removes all vector chunks associated with a specific document source filename.
    def delete_by_source(self, source: str) -> None:
        if source:
            self.collection.delete(where={"source": {"$eq": source}})

    # Returns the total number of vector embeddings currently indexed in the collection.
    def count(self) -> int:
        return self.collection.count()

    # Drops and recreates the collection to wipe all stored vector records.
    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

