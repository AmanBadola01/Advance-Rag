"""
Unit Tests for Parent-Child Context Expansion Retriever.
Verifies child-level search, parent-level context expansion, deduplication, and LCEL adapter.
"""

import uuid
import pytest
import chromadb

from app.models.schemas import RetrievalFilter
from app.services.chunking.parent_child import ParentChildChunker
from app.services.retrieval.base import LocalMiniLMEmbedding
from app.services.retrieval.basic_vector import BasicVectorRetriever
from app.services.retrieval.parent_child import ParentChildRetriever


@pytest.fixture
def clean_pc_retriever():
    client = chromadb.EphemeralClient()
    embed_fn = LocalMiniLMEmbedding()
    coll_name = f"test_pc_coll_{uuid.uuid4().hex[:8]}"
    vector = BasicVectorRetriever(client=client, embedding_function=embed_fn, collection_name=coll_name)
    chunker = ParentChildChunker(parent_chunk_size=300, parent_chunk_overlap=30, child_chunk_size=60, child_chunk_overlap=10)
    return ParentChildRetriever(child_retriever=vector, chunker=chunker)


def test_parent_child_ingest_and_counts(clean_pc_retriever):
    retriever = clean_pc_retriever
    text = (
        "Artificial intelligence and machine learning have evolved rapidly. "
        "Deep neural networks power language models, vision systems, and autonomous vehicles. "
        "Reinforcement learning trains agents to achieve complex goals through reward optimization. "
        "Supervised learning remains the bedrock of modern commercial classification systems."
    )

    counts = retriever.add_document(text=text, doc_id="doc_ai", source="ai_overview.txt")
    assert counts["parent_chunks_added"] >= 1
    assert counts["child_chunks_added"] >= 1

    total_counts = retriever.count()
    assert total_counts["parent_chunks"] == counts["parent_chunks_added"]
    assert total_counts["child_chunks"] == counts["child_chunks_added"]


def test_parent_context_expansion(clean_pc_retriever):
    retriever = clean_pc_retriever

    # Large text covering a full tutorial
    text = (
        "SECTION 1: INTRODUCTION TO POSTGRESQL. "
        "PostgreSQL is a powerful open-source object-relational database system. "
        "It provides high reliability, robust feature sets, and extensive extensibility. "
        "SECTION 2: RESOLVING CRITICAL MIGRATION ERROR E-4021. "
        "When migration failure error code E-4021 occurs, the database transaction has timed out. "
        "To fix error E-4021, increase statement_timeout to 60000ms in postgresql.conf and restart. "
        "SECTION 3: BACKUP AND RECOVERY PROCEDURES. "
        "Always create a full pg_dump backup before initiating schema migrations."
    )

    retriever.add_document(text=text, doc_id="doc_pg", source="pg_guide.txt")

    # Search specifically for the error code
    results = retriever.retrieve("How do I fix error E-4021?", top_k=1, return_parent_context=True)

    assert len(results) == 1
    retrieved = results[0]

    # Verify that the returned text is the full parent (includes SECTION 1 or surrounding context), not just the short child
    assert "SECTION" in retrieved.text
    assert "statement_timeout" in retrieved.text
    assert retrieved.strategy == "parent_child"
    assert retrieved.metadata["retrieval_mode"] == "parent_context_expansion"
    assert "matched_child_id" in retrieved.metadata
    assert "E-4021" in retrieved.metadata["matched_child_snippet"]


def test_parent_deduplication(clean_pc_retriever):
    retriever = clean_pc_retriever

    # A single large parent document that contains multiple related sub-points
    text = (
        "Kubernetes container orchestration guide. "
        "Pod deployment strategies include RollingUpdate and Recreate. "
        "Service discovery routes network traffic across pods using ClusterIP, NodePort, and LoadBalancer. "
        "Persistent Volumes provide durable storage independent of individual pod lifecycles. "
        "ConfigMaps and Secrets decouple configuration artifacts from container image content. "
        "Horizontal Pod Autoscalers dynamically adjust replica counts based on observed CPU utilization."
    )

    retriever.add_document(text=text, doc_id="doc_k8s", source="k8s.txt")

    # The query mentions both pods and services, matching multiple child chunks from the SAME parent
    results = retriever.retrieve("Kubernetes pods and services deployment", top_k=5, return_parent_context=True)

    # Even though multiple children matched, the parent should appear only ONCE
    parent_chunk_ids = [r.chunk_id for r in results]
    assert len(parent_chunk_ids) == len(set(parent_chunk_ids))
    assert len(results) == 1


def test_raw_child_retrieval_flag(clean_pc_retriever):
    retriever = clean_pc_retriever
    text = "Short child chunks offer sharp search. Large parent chunks give rich context."
    retriever.add_document(text=text, doc_id="doc_flag", source="flag.txt")

    # Setting return_parent_context=False should return raw child chunks
    child_results = retriever.retrieve("sharp search", top_k=2, return_parent_context=False)
    assert len(child_results) >= 1
    assert child_results[0].parent_id is not None


def test_metadata_filtering(clean_pc_retriever):
    retriever = clean_pc_retriever
    retriever.add_document(text="Document from staging environment.", doc_id="d1", source="staging.txt")
    retriever.add_document(text="Document from production environment.", doc_id="d2", source="prod.txt")

    filters = RetrievalFilter(source="staging.txt")
    results = retriever.retrieve("environment", top_k=5, filters=filters)

    assert len(results) == 1
    assert results[0].source == "staging.txt"


def test_langchain_lcel_adapter(clean_pc_retriever):
    retriever = clean_pc_retriever
    retriever.add_document(
        text="LangChain Expression Language LCEL simplifies production RAG workflows.",
        doc_id="d_lcel",
        source="lcel.txt",
    )

    lc_retriever = retriever.as_langchain_retriever(top_k=1)
    docs = lc_retriever.invoke("What does LCEL simplify?")

    assert len(docs) == 1
    assert "production RAG workflows" in docs[0].page_content
    assert docs[0].metadata["source"] == "lcel.txt"
    assert "matched_child_id" in docs[0].metadata


def test_delete_by_source(clean_pc_retriever):
    retriever = clean_pc_retriever
    retriever.add_document(text="Doc to delete.", doc_id="d1", source="delete_me.txt")
    retriever.add_document(text="Doc to retain.", doc_id="d2", source="keep_me.txt")

    assert retriever.count()["parent_chunks"] == 2

    retriever.delete_by_source("delete_me.txt")
    assert retriever.count()["parent_chunks"] == 1

    results = retriever.retrieve("Doc", top_k=5)
    assert all(r.source != "delete_me.txt" for r in results)

