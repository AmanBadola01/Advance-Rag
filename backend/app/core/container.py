"""
Service Container & Dependency Provider.
Initializes persistent ChromaDB, BM25, Hybrid, Re-Ranker, and Parent-Child retrievers,
and exposes shared singletons across FastAPI route handlers.
"""

import os
from typing import Dict, Any, Optional
import chromadb

from app.services.retrieval.base import LocalMiniLMEmbedding
from app.services.retrieval.basic_vector import BasicVectorRetriever
from app.services.retrieval.bm25 import BM25Retriever
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reranker import HybridRerankRetriever, CrossEncoderReranker
from app.services.retrieval.parent_child import ParentChildRetriever
from app.services.ingestion import DocumentIngestionService
from app.services.rag_chain import RAGChainService
from app.services.pipeline_tracer import pipeline_tracer


class ServiceContainer:
    """
    Central Service Container:
    Manages lifecycle of vector databases, lexical search indexes,
    document stores, ingestion services, and RAG orchestrators.
    """

    # Initializes storage directories, ChromaDB persistence client, embedding models, and retrievers.
    def __init__(self, db_path: str = "./backend/data/chroma_db"):
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path

        # 1. Initialize Vector Database & Embeddings
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_fn = LocalMiniLMEmbedding()
        self.vector_retriever = BasicVectorRetriever(
            client=self.chroma_client,
            embedding_function=self.embedding_fn,
            collection_name="production_rag_chunks",
        )

        # 2. Initialize Lexical BM25 & Hybrid Engine
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
        )

        # 3. Initialize Cross-Encoder Re-Ranker & Parent-Child Engines
        self.cross_encoder = CrossEncoderReranker()
        self.hybrid_rerank_retriever = HybridRerankRetriever(
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.cross_encoder,
            candidate_pool_size=20,
        )
        self.parent_child_retriever = ParentChildRetriever(
            child_retriever=self.vector_retriever,
        )

        # 4. In-Memory Raw Document Registry (Stores extracted text for on-demand A/B custom chunking)
        self.raw_documents_store: Dict[str, Dict[str, Any]] = {}

        # 5. Document Ingestion Service
        self.ingestion_service = DocumentIngestionService(
            hybrid_retriever=self.hybrid_retriever,
            parent_child_retriever=self.parent_child_retriever,
        )

        # 6. End-to-End RAG Chain Service
        self.rag_chain_service = RAGChainService(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            hybrid_retriever=self.hybrid_retriever,
            hybrid_rerank_retriever=self.hybrid_rerank_retriever,
            parent_child_retriever=self.parent_child_retriever,
            tracer=pipeline_tracer,
        )

    # Returns the global shared RAG chain service instance.
    def get_rag_chain_service(self) -> RAGChainService:
        return self.rag_chain_service

    # Returns the global document ingestion service instance.
    def get_ingestion_service(self) -> DocumentIngestionService:
        return self.ingestion_service

    # Records an ingested document into the raw document store for future A/B re-chunking.
    def register_raw_document(self, doc_id: str, filename: str, pages: list, strategy: str) -> None:
        self.raw_documents_store[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "pages": pages,
            "strategy": strategy,
        }

    # Retrieves raw document pages by document ID for dynamic on-demand chunking.
    def get_raw_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.raw_documents_store.get(doc_id)

    # Cascades document deletion across ChromaDB, BM25, and raw document registry.
    def delete_document(self, source: str) -> None:
        self.vector_retriever.delete_by_source(source)
        self.bm25_retriever.delete_by_source(source)
        self.parent_child_retriever.delete_by_source(source)

        to_remove = [k for k, v in self.raw_documents_store.items() if v.get("filename") == source]
        for k in to_remove:
            del self.raw_documents_store[k]

    # Clears all databases, indexes, and document registries across the platform.
    def clear_all(self) -> None:
        self.vector_retriever.clear()
        self.bm25_retriever.clear()
        self.parent_child_retriever.clear()
        self.raw_documents_store.clear()
        pipeline_tracer.clear()


# Application-wide singleton container instance
container = ServiceContainer()

