import pytest
from app.services.chunking.parent_child import ParentChildChunker


def test_parent_child_initialization():
    chunker = ParentChildChunker(parent_chunk_size=1000, child_chunk_size=200)
    assert chunker.parent_chunk_size == 1000
    assert chunker.child_chunk_size == 200


def test_parent_child_linkage_and_metadata():
    chunker = ParentChildChunker(parent_chunk_size=400, child_chunk_size=80)
    
    # Text sufficient to generate multiple parents and children
    paragraph = (
        "Parent-child chunking resolves the fundamental tradeoff between retrieval precision and synthesis context. "
        "Small 200-token child chunks produce dense, highly discriminative embeddings for vector search. "
        "Meanwhile, the larger 1000-token parent chunk preserves complete conversational and topical continuity. "
    )
    sample_text = paragraph * 15

    result = chunker.split_into_parent_child(
        sample_text,
        doc_id="pc_test",
        source="rag_architecture.pdf",
        page=1,
        file_type="pdf",
        custom_metadata={"topic": "Information Retrieval"}
    )

    parents = result.parents
    children = result.children

    assert len(parents) >= 2
    assert len(children) >= len(parents)

    # Map parent IDs
    parent_ids = {p.chunk_id for p in parents}

    # Verify parent properties
    for p in parents:
        assert p.metadata.is_parent is True
        assert p.metadata.parent_id is None
        assert p.metadata.strategy == "parent_child"
        assert p.metadata.source == "rag_architecture.pdf"
        assert p.metadata.token_count <= 420

    # Verify child properties and linkage
    for c in children:
        assert c.metadata.is_parent is False
        assert c.metadata.parent_id in parent_ids, f"Child {c.chunk_id} parent_id {c.metadata.parent_id} not in parents"
        assert c.metadata.strategy == "parent_child"
        assert c.metadata.token_count <= 90


def test_parent_resolution_helper():
    chunker = ParentChildChunker(parent_chunk_size=300, child_chunk_size=60)
    text = (
        "Section 1 introduces high-dimensional vector spaces and cosine distance metrics. "
        "Section 2 explores lexical indexing with Okapi BM25 and inverted index structures. "
    ) * 8

    result = chunker.split_into_parent_child(text, doc_id="lookup_doc")
    parent_store = {p.chunk_id: p for p in result.parents}

    # Simulate retrieval of first child chunk
    retrieved_child = result.children[0]
    resolved_parent = chunker.resolve_parent(retrieved_child, parent_store)

    assert resolved_parent.metadata.is_parent is True
    assert resolved_parent.chunk_id == retrieved_child.metadata.parent_id
    assert retrieved_child.text in resolved_parent.text
    assert resolved_parent.metadata.token_count >= retrieved_child.metadata.token_count


def test_parent_child_empty_text():
    chunker = ParentChildChunker()
    result = chunker.split_into_parent_child("")
    assert result.parents == []
    assert result.children == []


def test_parent_child_documents_batch():
    chunker = ParentChildChunker(parent_chunk_size=200, child_chunk_size=50)
    docs = [
        {"text": "Page one provides introduction to dense vector models.", "metadata": {"source": "book.pdf", "page": 1}},
        {"text": "Page two covers reciprocal rank fusion algorithms in detail.", "metadata": {"source": "book.pdf", "page": 2}}
    ]

    result = chunker.split_documents(docs, doc_id="batch_book")
    assert len(result.parents) >= 2
    assert len(result.children) >= 2

    parent_ids = {p.chunk_id for p in result.parents}
    for c in result.children:
        assert c.metadata.parent_id in parent_ids

