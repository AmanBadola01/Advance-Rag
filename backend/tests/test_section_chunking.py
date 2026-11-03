import pytest
from app.services.chunking_strategies import SectionBasedChunker


def test_section_chunker_initialization():
    chunker = SectionBasedChunker(max_section_tokens=600)
    assert chunker.max_section_tokens == 600
    assert len(chunker.headers_to_split_on) == 3


def test_section_chunker_markdown_headers():
    chunker = SectionBasedChunker(max_section_tokens=500)
    markdown_doc = """# Annual Report 2025
This is the general intro to the annual report.

## Financial Summary
In Q3 2025, overall net revenue increased by 24% year-over-year. Operating income stood at $4.2M.

## Product Roadmap
We launched three core autonomous agent capabilities and expanded enterprise RAG integrations.
"""
    chunks = chunker.split_text(markdown_doc, doc_id="annual_doc", source="report.md", file_type="md")
    
    assert len(chunks) == 3
    sections = [c.metadata.section for c in chunks]
    assert "Annual Report 2025" in sections
    assert "Financial Summary" in sections
    assert "Product Roadmap" in sections

    for c in chunks:
        assert c.metadata.strategy == "section"
        assert c.metadata.source == "report.md"
        assert c.metadata.file_type == "md"
        assert c.metadata.token_count > 0
        assert c.chunk_id.startswith("annual_doc_section_")


def test_section_chunker_plain_text_normalization():
    chunker = SectionBasedChunker(max_section_tokens=500)
    plain_doc = """1.0 System Architecture
The platform is designed around a microservice layout with LangChain LCEL chains and FastAPI.

2.0 Security Model
All data access enforces RBAC policies and sandboxed code execution environments.

SECTION 3: Deployment Strategy
The backend is packaged into Docker containers and orchestrated with Docker Compose.
"""
    chunks = chunker.split_text(plain_doc, doc_id="arch_doc", source="architecture.txt", file_type="txt")
    
    assert len(chunks) == 3
    sections = [c.metadata.section for c in chunks]
    assert any("System Architecture" in s for s in sections)
    assert any("Security Model" in s for s in sections)
    assert any("Deployment Strategy" in s for s in sections)


def test_section_chunker_large_section_subsplit():
    # Set small max_section_tokens to force sub-splitting
    chunker = SectionBasedChunker(max_section_tokens=60, sub_split_large_sections=True)
    long_content = (
        "## Deep Learning Overview\n"
        "Modern deep learning architectures leverage transformers for natural language understanding. "
        "Attention mechanisms compute pairwise token interactions across bidirectional context. "
        "Self-attention layers are followed by feed-forward networks and layer normalization. "
    ) * 4

    chunks = chunker.split_text(long_content, doc_id="dl_doc", source="ai_paper.md")
    assert len(chunks) > 1

    for c in chunks:
        assert c.metadata.section == "Deep Learning Overview"
        assert c.metadata.strategy == "section"
        assert c.metadata.token_count <= 70
        assert "part_index" in c.metadata.custom_metadata


def test_section_chunker_empty_and_whitespace():
    chunker = SectionBasedChunker()
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\n   ") == []


def test_section_chunker_documents_batch():
    chunker = SectionBasedChunker(max_section_tokens=500)
    docs = [
        {
            "text": "# Executive Summary\nKey findings of the audit.",
            "metadata": {"source": "audit.pdf", "page": 1}
        },
        {
            "text": "## Compliance Findings\nZero major non-conformances identified.",
            "metadata": {"source": "audit.pdf", "page": 2}
        }
    ]
    chunks = chunker.split_documents(docs, doc_id="audit_batch")
    assert len(chunks) == 2
    assert chunks[0].metadata.section == "Executive Summary"
    assert chunks[0].metadata.page == 1
    assert chunks[1].metadata.section == "Compliance Findings"
    assert chunks[1].metadata.page == 2

