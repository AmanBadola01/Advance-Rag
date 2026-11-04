import pytest
import chromadb
from app.models.schemas import RetrievalFilter
from app.services.chunking import RecursiveChunker
from app.services.retrieval.basic_vector import BasicVectorRetriever
from app.services.retrieval.bm25 import BM25Retriever
from app.services.retrieval.hybrid import HybridRetriever


@pytest.fixture
def hybrid_suite():
    """Create isolated in-memory HybridRetriever with Vector and BM25 stores."""
    client = chromadb.Client()
    vec_retriever = BasicVectorRetriever(
        collection_name="test_hybrid_collection",
        client=client,
    )
    bm25_retriever = BM25Retriever()
    hybrid = HybridRetriever(
        vector_retriever=vec_retriever,
        bm25_retriever=bm25_retriever,
        semantic_weight=0.5,
        bm25_weight=0.5,
        rrf_k=60,
    )
    yield hybrid
    hybrid.clear()


def test_bm25_exact_code_matching():
    retriever = BM25Retriever()
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

    doc1 = "The system encountered critical failure error code E-4021 during database migration."
    doc2 = "General troubleshooting tips for network connection timeouts."

    c1 = chunker.split_text(doc1, doc_id="d1", source="err.txt")[0]
    c2 = chunker.split_text(doc2, doc_id="d2", source="net.txt")[0]

    retriever.add_chunks([c1, c2])

    results = retriever.retrieve("E-4021", top_k=2)
    assert len(results) == 1
    assert "E-4021" in results[0].text
    assert results[0].bm25_score > 0
    assert results[0].score == 1.0  # Max score normalized to 1.0


def test_hybrid_add_chunks_and_count(hybrid_suite):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    c1 = chunker.split_text("Python FastAPI backend implementation", doc_id="d1", source="fastapi.txt")[0]
    c2 = chunker.split_text("Next.js React frontend dashboard", doc_id="d2", source="next.txt")[0]

    counts = hybrid_suite.add_chunks([c1, c2])
    assert counts["vector_added"] == 2
    assert counts["bm25_added"] == 2
    assert hybrid_suite.count()["vector_chunks"] == 2
    assert hybrid_suite.count()["bm25_chunks"] == 2


def test_hybrid_rrf_boost_for_dual_matches(hybrid_suite):
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=20)

    # Chunk A has both conceptual relevance AND exact keyword
    chunk_a = (
        "Project Apollo: Q3 quarterly financial earnings report shows revenue growth of 24% "
        "and gross margins of 18% with net operating profit expanding."
    )
    # Chunk B has only general concept (financial overview) but no Apollo keyword
    chunk_b = (
        "General corporate annual performance review discusses revenue strategies, profit optimization, "
        "and long term shareholder returns."
    )
    # Chunk C has only the keyword 'Apollo' in unrelated context
    chunk_c = (
        "Project Apollo launch schedule details rocket propellants and lunar trajectory telemetry data."
    )

    chunks_a = chunker.split_text(chunk_a, doc_id="ca", source="apollo_fin.pdf")
    chunks_b = chunker.split_text(chunk_b, doc_id="cb", source="general_fin.pdf")
    chunks_c = chunker.split_text(chunk_c, doc_id="cc", source="apollo_space.pdf")

    hybrid_suite.add_chunks(chunks_a + chunks_b + chunks_c)

    # Query requires BOTH concept (financial revenue) and keyword (Apollo)
    results = hybrid_suite.retrieve(query="Project Apollo financial revenue growth", top_k=3)

    assert len(results) >= 2
    top_hit = results[0]

    # Chunk A must rank #1 because it scored high in both Dense Vector AND BM25
    assert "apollo_fin.pdf" in top_hit.source
    assert top_hit.rank == 1
    assert top_hit.score > 0  # Combined RRF score
    assert "dense_score" in top_hit.metadata
    assert "bm25_score" in top_hit.metadata
    assert top_hit.metadata["dense_rank"] is not None
    assert top_hit.metadata["bm25_rank"] is not None


def test_hybrid_weight_configuration(hybrid_suite):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    # Exact code match
    c1 = chunker.split_text("Serial identifier SKU-998822 in inventory warehouse.", doc_id="d1", source="inv.txt")[0]
    # Semantic match
    c2 = chunker.split_text("Stock storage facility containing assorted products.", doc_id="d2", source="stock.txt")[0]

    hybrid_suite.add_chunks([c1, c2])

    # 1. Heavily favor BM25 (90% BM25, 10% semantic)
    bm25_heavy = hybrid_suite.retrieve(
        query="SKU-998822 warehouse",
        top_k=2,
        semantic_weight=0.1,
        bm25_weight=0.9,
    )
    assert bm25_heavy[0].chunk_id == c1.chunk_id

    # 2. Heavily favor semantic (90% semantic, 10% BM25)
    semantic_heavy = hybrid_suite.retrieve(
        query="product storage facility",
        top_k=2,
        semantic_weight=0.9,
        bm25_weight=0.1,
    )
    assert semantic_heavy[0].chunk_id == c2.chunk_id


def test_hybrid_metadata_filtering(hybrid_suite):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    c1 = chunker.split_text("Q3 results", doc_id="d1", source="fin_2025.pdf", section="Q3")[0]
    c2 = chunker.split_text("Q3 results", doc_id="d2", source="fin_2024.pdf", section="Q3")[0]

    hybrid_suite.add_chunks([c1, c2])

    filtered = hybrid_suite.retrieve(
        query="Q3 results",
        top_k=5,
        filters=RetrievalFilter(source="fin_2025.pdf"),
    )
    assert len(filtered) == 1
    assert filtered[0].source == "fin_2025.pdf"


def test_hybrid_langchain_lcel_adapter(hybrid_suite):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    c = chunker.split_text("Ensemble hybrid retrieval for LangChain LCEL.", doc_id="d1", source="lcel.md")[0]
    hybrid_suite.add_chunks([c])

    lc_retriever = hybrid_suite.as_langchain_retriever(top_k=1)
    docs = lc_retriever.invoke("Ensemble hybrid retrieval")

    assert len(docs) == 1
    assert "Ensemble hybrid" in docs[0].page_content
    assert "score" in docs[0].metadata
    assert docs[0].metadata["source"] == "lcel.md"


def test_hybrid_delete_by_source(hybrid_suite):
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    c1 = chunker.split_text("File to remove", doc_id="d1", source="delete_me.txt")[0]
    c2 = chunker.split_text("File to preserve", doc_id="d2", source="preserve.txt")[0]

    hybrid_suite.add_chunks([c1, c2])
    assert hybrid_suite.count()["vector_chunks"] == 2
    assert hybrid_suite.count()["bm25_chunks"] == 2

    hybrid_suite.delete_by_source("delete_me.txt")
    assert hybrid_suite.count()["vector_chunks"] == 1
    assert hybrid_suite.count()["bm25_chunks"] == 1

    remaining = hybrid_suite.retrieve("File", top_k=5)
    assert len(remaining) == 1
    assert remaining[0].source == "preserve.txt"

