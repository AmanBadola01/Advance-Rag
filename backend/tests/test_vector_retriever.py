import pytest
import chromadb
from app.models.schemas import RetrievalFilter
from app.services.chunking import RecursiveChunker
from app.services.retrieval.basic_vector import BasicVectorRetriever


@pytest.fixture
def test_retriever():
    """Create isolated in-memory BasicVectorRetriever for each test."""
    client = chromadb.Client()
    retriever = BasicVectorRetriever(
        collection_name="test_vector_collection",
        client=client,
    )
    yield retriever
    retriever.clear()


def test_vector_retriever_initialization(test_retriever):
    assert test_retriever.collection_name == "test_vector_collection"
    assert test_retriever.count() == 0


def test_vector_retriever_add_and_retrieve(test_retriever):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    doc1 = "In Q3 2025, operating profit surged by 34% driven by enterprise cloud adoption and net margin expansion."
    doc2 = "The zero-trust security architecture requires mutual TLS authentication and strict RBAC authorization."

    chunks1 = chunker.split_text(doc1, doc_id="finance_doc", source="earnings.pdf", section="Financials")
    chunks2 = chunker.split_text(doc2, doc_id="security_doc", source="security.pdf", section="Security")

    test_retriever.add_chunks(chunks1 + chunks2)
    assert test_retriever.count() == len(chunks1) + len(chunks2)

    # Query finance concepts
    finance_results = test_retriever.retrieve(query="cloud revenue and operating margin growth", top_k=2)
    assert len(finance_results) > 0
    top_hit = finance_results[0]

    assert top_hit.source == "earnings.pdf"
    assert top_hit.section == "Financials"
    assert top_hit.score > 0.4
    assert top_hit.rank == 1


def test_vector_retriever_metadata_filtering(test_retriever):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    c1 = chunker.split_text("Revenue metrics report", doc_id="d1", source="report_a.pdf", section="Summary")[0]
    c2 = chunker.split_text("Revenue metrics breakdown", doc_id="d2", source="report_b.pdf", section="Details")[0]

    test_retriever.add_chunks([c1, c2])

    # Filter strictly by source 'report_a.pdf'
    filtered_results = test_retriever.retrieve(
        query="Revenue metrics",
        top_k=5,
        filters=RetrievalFilter(source="report_a.pdf"),
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].source == "report_a.pdf"

    # Filter strictly by section 'Details'
    section_results = test_retriever.retrieve(
        query="Revenue metrics",
        top_k=5,
        filters=RetrievalFilter(section="Details"),
    )
    assert len(section_results) == 1
    assert section_results[0].section == "Details"


def test_vector_retriever_langchain_lcel_adapter(test_retriever):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text(
        "LangChain LCEL provides unified Runnable protocols for pipelines.",
        doc_id="lc_doc",
        source="langchain.md"
    )
    test_retriever.add_chunks(chunks)

    # Adapt to LangChain retriever
    lc_retriever = test_retriever.as_langchain_retriever(top_k=1)
    docs = lc_retriever.invoke("LangChain Runnable protocols")

    assert len(docs) == 1
    assert "Runnable protocols" in docs[0].page_content
    assert docs[0].metadata["source"] == "langchain.md"
    assert "score" in docs[0].metadata


def test_vector_retriever_delete_by_source(test_retriever):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    c1 = chunker.split_text("Document to delete", doc_id="del_doc", source="temp_delete.pdf")[0]
    c2 = chunker.split_text("Document to keep", doc_id="keep_doc", source="keep.pdf")[0]

    test_retriever.add_chunks([c1, c2])
    assert test_retriever.count() == 2

    test_retriever.delete_by_source("temp_delete.pdf")
    assert test_retriever.count() == 1

    remaining = test_retriever.retrieve(query="Document", top_k=5)
    assert len(remaining) == 1
    assert remaining[0].source == "keep.pdf"


def test_vector_retriever_empty_query(test_retriever):
    assert test_retriever.retrieve("") == []
    assert test_retriever.retrieve("   ") == []

