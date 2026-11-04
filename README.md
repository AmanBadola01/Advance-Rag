# Production-Grade Advanced RAG Platform with LangChain

An enterprise-ready Retrieval-Augmented Generation (RAG) platform built with **FastAPI**, **LangChain (LCEL)**, **ChromaDB**, and **Next.js**.

This project implements advanced RAG methodologies including **Hybrid Search (Dense + BM25 with Reciprocal Rank Fusion)**, **Cross-Encoder Re-Ranking**, **Multi-Strategy Ingestion (Recursive, Semantic, Parent-Child, Section-Based)**, **Query Transformations (HyDE, Multi-Query, Decomposition)**, and **Quantitative RAG Evaluation (Faithfulness, Answer Relevancy, Context Precision, Context Recall)**.

---

## 🏗️ Architectural Overview

```
                      [ User Query ]
                            │
               [ Query Transformation / HyDE ]
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
 [ Dense Vector Search ]         [ Sparse BM25 Search ]
     (ChromaDB)                   (rank_bm25 Inverted Index)
            │                               │
            └───────────────┬───────────────┘
                            ▼
             [ Reciprocal Rank Fusion (RRF) ]
                (Top-20 Candidate Pool)
                            │
                            ▼
             [ Cross-Encoder Re-Ranker ]
          (ms-marco-MiniLM-L-6-v2 / Cohere)
                            │
                            ▼
             [ Top-5 High-Precision Chunks ]
            (Parent Resolution if Parent-Child)
                            │
                            ▼
             [ Context Assembly & Prompt ]
                            │
                            ▼
             [ LLM Generation & Citations ]
               (Gemini / Groq / OpenAI)
```

---

## 📁 Repository Structure

```text
advanced-rag-platform/
|-- backend/
|   |-- app/
|   |   |-- main.py                  # FastAPI application entry point
|   |   |-- config.py                # Environment & settings configuration
|   |   |-- routers/                 # API route handlers
|   |   |-- services/                # Core RAG engine & algorithms
|   |   |   |-- chunking/            # Modular chunking strategies
|   |   |   |   |-- base.py          # BaseChunker & shared token counter
|   |   |   |   |-- recursive.py     # Recursive character splitting
|   |   |   |   |-- section.py       # Section-based header splitting
|   |   |   |   |-- parent_child.py  # Parent-Child hierarchical splitting
|   |   |   |   |-- __init__.py      # Package export
|   |   |-- models/                  # Data models
|   |   |   |-- schemas.py           # Pydantic data contracts (ChunkMetadata, ProcessedChunk, ParentChildResult)
|   |   |   |-- database.py          # SQLAlchemy models (SQLite)
|   |   |-- data/                    # Evaluation datasets & test fixtures
|   |-- tests/                       # Automated test suites
|   |   |-- test_recursive_chunking.py    # Test suite for RecursiveChunker
|   |   |-- test_section_chunking.py      # Test suite for SectionBasedChunker
|   |   |-- test_parent_child_chunking.py # Test suite for ParentChildChunker
|   |-- requirements.txt             # Backend dependencies
|   |-- Dockerfile
|-- frontend/                        # Next.js 14+ UI with Tailwind & shadcn/ui (upcoming)
|-- sample_documents/                # Test documents (PDF, TXT, DOCX, MD)
|-- eval_dataset/                    # 20+ Q&A evaluation benchmark pairs
|-- .env.example                     # Environment template
|-- .gitignore                       # Git ignore configuration
|-- README.md                        # Documentation
```

---

## 🚀 Progress & Implemented Features

### Phase 1: Ingestion & Chunking Engine (In Progress)

- [x] **Project Scaffolding & Virtual Environment**: Python 3.12 environment initialized with LangChain LCEL modular packages.
- [x] **Data Contracts & Pydantic Schemas (`app/models/schemas.py`)**:
  - `ChunkMetadata`: Standardized schema for all chunk metadata (`chunk_id`, `chunk_index`, `strategy`, `source`, `page`, `file_type`, `token_count`, `char_count`, `section`, `parent_id`, `is_parent`, `custom_metadata`).
  - `ProcessedChunk`: Strict validation container for chunk content and metadata.
- [x] **Base Strategy Pattern (`app/services/chunking_strategies.py`)**:
  - `BaseChunker`: Centralized token counting using `tiktoken` (`cl100k_base` encoding) with fallback approximation, plus uniform chunk construction.
- [x] **Strategy 1: Recursive Character Splitting (`RecursiveChunker`)**:
  - Implements LangChain's `RecursiveCharacterTextSplitter`.
  - Configured for **500 tokens** chunk size and **50 tokens** overlap.
  - Hierarchical boundary splits: Paragraph (`\n\n`) $\rightarrow$ Sentence (`\n`) $\rightarrow$ Word (` `) $\rightarrow$ Character (`""`) to maintain clean contextual boundaries.
  - Supports single text strings (`split_text`) and multi-page document batches (`split_documents`).
- [x] **Strategy 2: Section-Based Chunking (`SectionBasedChunker`)**:
  - Implements LangChain's `MarkdownHeaderTextSplitter` targeting H1 (`#`), H2 (`##`), and H3 (`###`) hierarchies.
  - Automatic plain-text header normalization for structured reports (e.g. `1.0 Introduction`, `SECTION 2:`, `APPENDIX A`).
  - Automatically extracts and attaches `section` header metadata to every chunk for downstream metadata filtering.
  - Safe overflow protection: Sub-splits sections exceeding `max_section_tokens` (800 tokens) while preserving section tags across all parts.
- [x] **Strategy 3: Parent-Child Chunking (`ParentChildChunker`)**:
  - Implements two-tier hierarchical chunking:
    - **Child Chunks (~200 tokens)**: Small, dense chunks optimized for vector similarity search.
    - **Parent Chunks (~1000 tokens)**: Large chunks providing complete context window to the LLM during generation.
  - Bidirectional metadata linkage: Child chunks carry `parent_id` matching their parent's primary key (`is_parent=False`), while parent chunks are tagged with `is_parent=True`.
  - Includes `resolve_parent(child_chunk, parent_store)` resolver to automatically retrieve parent context when a child is matched.
- [x] **Automated Verification**:
  - `tests/test_recursive_chunking.py`: 6 unit tests covering initialization, boundary splitting, overlap validation, multi-document batching, and token limit bounds.
  - `tests/test_section_chunking.py`: 6 unit tests covering Markdown headers, plain text normalization, section metadata extraction, large section sub-splitting, and multi-document batches.
  - `tests/test_parent_child_chunking.py`: 5 unit tests covering parent-child token bounds, ID linkage integrity, parent resolution lookup, empty text handling, and multi-page batching.
  - **Verdict**: 17/17 tests passing.
- [x] **Gravitas Reliability Runtime Integration**: Enforcing ground-before-mutating, scoped changes, and evidence-backed verification.

---

## 🧪 Running the Tests

To verify the chunking implementations:

```bash
# 1. Activate backend environment
cd backend
# On Windows:
.\venv\Scripts\Activate.ps1

# 2. Run pytest suite
$env:PYTHONPATH = "backend"
pytest backend/tests/ -v
```

Expected output:
```text
backend/tests/test_recursive_chunking.py::test_recursive_chunker_initialization PASSED
backend/tests/test_recursive_chunking.py::test_recursive_chunker_basic_split PASSED
backend/tests/test_recursive_chunking.py::test_recursive_chunker_overlap PASSED
backend/tests/test_recursive_chunking.py::test_recursive_chunker_empty_and_whitespace PASSED
backend/tests/test_recursive_chunking.py::test_recursive_chunker_realistic_500_tokens PASSED
backend/tests/test_recursive_chunking.py::test_recursive_chunker_documents_batch PASSED
backend/tests/test_section_chunking.py::test_section_chunker_initialization PASSED
backend/tests/test_section_chunking.py::test_section_chunker_markdown_headers PASSED
backend/tests/test_section_chunking.py::test_section_chunker_plain_text_normalization PASSED
backend/tests/test_section_chunking.py::test_section_chunker_large_section_subsplit PASSED
backend/tests/test_section_chunking.py::test_section_chunker_empty_and_whitespace PASSED
backend/tests/test_section_chunking.py::test_section_chunker_documents_batch PASSED
============================== 12 passed ==============================
```

---

## 🗺️ Next Steps

1. **Chunking Strategy 3**: Parent-Child Chunking (200-token retrieval child chunks linked to 1000-token parent context chunks).
2. **Chunking Strategy 4**: Semantic Chunking (similarity threshold-based splitting).
3. **Document Ingestion Pipeline**: LangChain loaders for PDF, TXT, DOCX, and Markdown with metadata enrichment.
4. **ChromaDB & BM25 Storage**: Storing all simultaneous chunking strategies with query-time filtering.

