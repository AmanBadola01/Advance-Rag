"""
Unit Tests for Cross-Encoder Re-Ranker & Hybrid + Re-Ranking Retriever.
Tests two-stage candidate retrieval, score calibration, answer prioritization, and LCEL adapter.
"""

import uuid
import pytest
import chromadb
from app.models.schemas import RetrievalFilter
from app.services.chunking.recursive import RecursiveChunker
from app.services.retrieval.base import LocalMiniLMEmbedding
from app.services.retrieval.basic_vector import BasicVectorRetriever
from app.services.retrieval.bm25 import BM25Retriever
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reranker import (
    CrossEncoderReranker,
    HybridRerankRetriever,
    sigmoid_score,
)


@pytest.fixture
def clean_hybrid_reranker():
    client = chromadb.EphemeralClient()
    embed_fn = LocalMiniLMEmbedding()
    coll_name = f"test_reranker_coll_{uuid.uuid4().hex[:8]}"
    vector = BasicVectorRetriever(client=client, embedding_function=embed_fn, collection_name=coll_name)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25)
    reranker = CrossEncoderReranker()
    two_stage = HybridRerankRetriever(hybrid_retriever=hybrid, reranker=reranker, candidate_pool_size=10)
    return two_stage


def test_sigmoid_score_bounds():
    assert 0.0 <= sigmoid_score(0.0) <= 1.0
    assert sigmoid_score(0.0) == 0.5
    assert sigmoid_score(10.0) > 0.99
    assert sigmoid_score(-10.0) < 0.01
    # Clamping tests for extreme values
    assert 0.0 <= sigmoid_score(100.0) <= 1.0
    assert 0.0 <= sigmoid_score(-100.0) <= 1.0


def test_cross_encoder_standalone_prioritization():
    reranker = CrossEncoderReranker()
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=10)

    # Distractor: repeats query words "error" and "database" multiple times without answering
    doc_distractor = "Database database error error log. Migration database system error reference index."
    # Direct answer: clearly explains the actual fix
    doc_answer = "To resolve the migration failure, restart the PostgreSQL service and run vacuum analyze."

    c_distractor = chunker.split_text(doc_distractor, doc_id="d1", source="index.txt")[0]
    c_answer = chunker.split_text(doc_answer, doc_id="d2", source="guide.txt")[0]

    from app.models.schemas import RetrievedChunk
    cand1 = RetrievedChunk(
        chunk_id=c_distractor.chunk_id,
        text=c_distractor.text,
        score=0.95,
        rank=1,
        strategy="hybrid",
        source=c_distractor.metadata.source,
        page=1,
    )
    cand2 = RetrievedChunk(
        chunk_id=c_answer.chunk_id,
        text=c_answer.text,
        score=0.70,
        rank=2,
        strategy="hybrid",
        source=c_answer.metadata.source,
        page=1,
    )

    # Even though cand1 came in with higher initial score, the reranker should prioritize the direct answer
    reranked = reranker.rerank(
        query="How do I fix the database migration failure?",
        candidates=[cand1, cand2],
        top_k=2,
    )

    assert len(reranked) == 2
    assert reranked[0].chunk_id == cand2.chunk_id
    assert "restart the PostgreSQL service" in reranked[0].text
    assert reranked[0].rank == 1
    assert reranked[0].metadata["pre_rerank_rank"] == 2
    assert "rerank_score" in reranked[0].metadata


def test_two_stage_hybrid_reranker_pipeline(clean_hybrid_reranker):
    retriever = clean_hybrid_reranker
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

    docs = [
        ("The quick brown fox jumps over the lazy dog.", "nature.txt"),
        ("To reset the user password, click Forgot Password on the login page.", "auth.txt"),
        ("System architecture diagram and cloud infrastructure topology.", "arch.txt"),
        ("Billing and subscription invoice download instructions.", "billing.txt"),
    ]

    all_chunks = []
    for idx, (text, src) in enumerate(docs):
        chunks = chunker.split_text(text, doc_id=f"doc_{idx}", source=src)
        all_chunks.extend(chunks)

    retriever.add_chunks(all_chunks)

    results = retriever.retrieve("How can I reset my password?", top_k=2)

    assert len(results) == 2
    assert results[0].source == "auth.txt"
    assert "click Forgot Password" in results[0].text
    assert results[0].metadata["retrieval_mode"] == "hybrid_rerank"
    assert 0.0 <= results[0].score <= 1.0


def test_two_stage_metadata_filtering(clean_hybrid_reranker):
    retriever = clean_hybrid_reranker
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

    c1 = chunker.split_text("Configuration guide for production servers.", doc_id="d1", source="prod.txt")[0]
    c2 = chunker.split_text("Configuration guide for staging servers.", doc_id="d2", source="staging.txt")[0]

    retriever.add_chunks([c1, c2])

    filters = RetrievalFilter(source="staging.txt")
    results = retriever.retrieve("server configuration", top_k=5, filters=filters)

    assert len(results) == 1
    assert results[0].source == "staging.txt"
    assert "staging servers" in results[0].text


def test_two_stage_langchain_lcel_adapter(clean_hybrid_reranker):
    retriever = clean_hybrid_reranker
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

    c = chunker.split_text("LangChain Expression Language LCEL enables streaming and async pipelines.", doc_id="d1", source="lcel.txt")[0]
    retriever.add_chunks([c])

    lc_retriever = retriever.as_langchain_retriever(top_k=1)
    docs = lc_retriever.invoke("What does LCEL enable?")

    assert len(docs) == 1
    assert "LangChain Expression Language" in docs[0].page_content
    assert docs[0].metadata["source"] == "lcel.txt"
    assert docs[0].metadata["retrieval_mode"] == "hybrid_rerank"


def test_two_stage_delete_by_source(clean_hybrid_reranker):
    retriever = clean_hybrid_reranker
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

    c1 = chunker.split_text("Doc 1 to delete.", doc_id="d1", source="del.txt")[0]
    c2 = chunker.split_text("Doc 2 to keep.", doc_id="d2", source="keep.txt")[0]

    retriever.add_chunks([c1, c2])
    counts = retriever.count()
    assert counts["vector_chunks"] == 2

    retriever.delete_by_source("del.txt")
    post_counts = retriever.count()
    assert post_counts["vector_chunks"] == 1
    assert post_counts["bm25_chunks"] == 1

    results = retriever.retrieve("Doc 1", top_k=5)
    assert all(r.source != "del.txt" for r in results)

