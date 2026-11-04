import pytest
from app.services.chunking.recursive import RecursiveChunker


def test_recursive_chunker_initialization():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 50
    assert chunker.tokenizer is not None


def test_recursive_chunker_basic_split():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    
    # 3 distinct paragraphs of text
    p1 = "LangChain provides composable abstractions for building modern RAG pipelines. " * 3
    p2 = "Hybrid search combines dense semantic embeddings with sparse BM25 lexical retrieval. " * 3
    p3 = "Cross-encoder re-ranking calculates direct query-chunk relevance scores to boost precision. " * 3
    sample_text = f"{p1}\n\n{p2}\n\n{p3}"

    chunks = chunker.split_text(
        sample_text,
        doc_id="test_doc_01",
        source="rag_guide.txt",
        page=1,
        file_type="txt",
        custom_metadata={"author": "Antigravity", "domain": "AI"}
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.metadata.strategy == "recursive"
        assert chunk.metadata.source == "rag_guide.txt"
        assert chunk.metadata.page == 1
        assert chunk.metadata.file_type == "txt"
        assert chunk.metadata.custom_metadata["author"] == "Antigravity"
        assert chunk.metadata.token_count <= 55  # Allows small boundary tolerance
        assert chunk.metadata.char_count == len(chunk.text)
        assert chunk.chunk_id.startswith("test_doc_01_recursive_")


def test_recursive_chunker_overlap():
    chunker = RecursiveChunker(chunk_size=40, chunk_overlap=15)
    text = "The quick brown fox jumps over the lazy dog repeatedly to demonstrate chunk overlap in document processing. " * 5
    
    chunks = chunker.split_text(text, doc_id="overlap_doc", source="fox.txt")
    assert len(chunks) > 1

    # Check that the beginning phrase of subsequent chunk is present in the previous chunk (overlap)
    overlap_found = False
    for i in range(len(chunks) - 1):
        lead_phrase = " ".join(chunks[i + 1].text.split()[:4])
        if lead_phrase in chunks[i].text:
            overlap_found = True
            break
    assert overlap_found, "Expected overlap between consecutive chunks"


def test_recursive_chunker_empty_and_whitespace():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\n   ") == []


def test_recursive_chunker_realistic_500_tokens():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    
    # Generate ~1500 tokens of realistic text
    paragraph = (
        "Advanced Retrieval-Augmented Generation (RAG) overcomes critical limitations of basic vector search. "
        "By coupling semantic embeddings with BM25 lexical indexing, hybrid retrieval captures both contextual "
        "semantics and exact keyword occurrences such as model serial numbers or error codes. "
        "Cross-encoders subsequently re-rank the retrieved pool to elevate truly pertinent contexts.\n\n"
    )
    long_document = paragraph * 30  # ~1500 tokens

    chunks = chunker.split_text(long_document, doc_id="doc_500", source="rag_advancement.txt")
    assert len(chunks) >= 3

    for idx, c in enumerate(chunks):
        assert c.metadata.token_count <= 505, f"Chunk {idx} exceeded 500 token limit: {c.metadata.token_count}"
        assert c.metadata.strategy == "recursive"
        assert len(c.text) > 0



def test_recursive_chunker_documents_batch():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    docs = [
        {"text": "Page 1 content about vector search and embeddings.", "metadata": {"source": "manual.pdf", "page": 1}},
        {"text": "Page 2 content about BM25 sparse keyword indices and rank fusion.", "metadata": {"source": "manual.pdf", "page": 2}}
    ]
    chunks = chunker.split_documents(docs, doc_id="manual_doc")
    assert len(chunks) >= 2
    pages = [c.metadata.page for c in chunks]
    assert 1 in pages
    assert 2 in pages
