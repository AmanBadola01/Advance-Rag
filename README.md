# Production-Grade Advanced RAG Platform with LangChain

An enterprise-ready Retrieval-Augmented Generation (RAG) platform built with **FastAPI**, **LangChain (LCEL)**, **ChromaDB**, and **Next.js**.

This project implements advanced RAG methodologies including **Hybrid Search (Dense + BM25 with Reciprocal Rank Fusion)**, **Cross-Encoder Re-Ranking**, **Multi-Strategy Ingestion (Recursive, Section-Based, Parent-Child)**, **Context Expansion**, and **Quantitative RAG Evaluation**.

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
          (ms-marco-MiniLM-L-6-v2 / Offline)
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
|   |   |-- main.py                       # FastAPI application entry point
|   |   |-- config.py                     # Environment & settings configuration
|   |   |-- api/                          # API route handlers (documents, query, chunks)
|   |   |-- services/                     # Core RAG engine & algorithms
|   |   |   |-- chunking/                 # Modular chunking strategies
|   |   |   |   |-- base.py               # BaseChunker & shared tiktoken counter
|   |   |   |   |-- recursive.py          # Recursive character splitting (500 tokens, 50 overlap)
|   |   |   |   |-- section.py            # Section-based header splitting (Markdown & Plain Text)
|   |   |   |   |-- parent_child.py       # Parent-Child hierarchical splitting (1000/200 tokens)
|   |   |   |   |-- __init__.py           # Package export
|   |   |   |-- retrieval/                # Modular retrieval strategies
|   |   |   |   |-- base.py               # ChromaDB embedding wrapper & boolean filter builder
|   |   |   |   |-- basic_vector.py       # Dense semantic vector search (ChromaDB + all-MiniLM-L6-v2)
|   |   |   |   |-- bm25.py               # Sparse keyword search (Lucene-smoothed Okapi BM25)
|   |   |   |   |-- hybrid.py             # Hybrid search (Dense + BM25 via RRF k=60)
|   |   |   |   |-- reranker.py           # Two-stage cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
|   |   |   |   |-- parent_child.py       # Context expansion retriever ("Search small, read big")
|   |   |   |   |-- __init__.py           # Package export
|   |   |   |-- ingestion.py              # Multi-format document parser (PDF, DOCX, TXT, MD)
|   |   |   |-- pipeline_tracer.py        # Latency (ms), token usage & execution tracing
|   |   |   |-- rag_chain.py              # LangChain LCEL RAG prompt & generation chain
|   |   |-- models/                       # Data models & schemas
|   |   |   |-- schemas.py                # Pydantic data contracts (RetrievedChunk, ProcessedChunk, etc.)
|   |-- requirements.txt                  # Backend dependencies
|-- frontend/                             # Next.js 14+ UI with Tailwind & shadcn/ui
|-- .gitignore                            # Git ignore configuration
|-- README.md                             # Project documentation
```

---

## 🚀 Implemented Features & Technical Architecture

### 1. Chunking Strategies (`app/services/chunking/`)

- [x] **Base Strategy Pattern (`base.py`)**: Centralized token counting using `tiktoken` (`cl100k_base` encoding) with fallback character approximation.
- [x] **Recursive Character Splitting (`recursive.py`)**: 500-token chunks with 50-token overlap across hierarchical boundaries (`\n\n` $\rightarrow$ `\n` $\rightarrow$ ` ` $\rightarrow$ `""`).
- [x] **Section-Based Chunking (`section.py`)**: Preserves Markdown headers (H1/H2/H3) and plain-text sections (`1.0 Introduction`, `SECTION:`) with automatic sub-splitting for oversized sections.
- [x] **Parent-Child Chunking (`parent_child.py`)**: Generates 1000-token parent context chunks and 200-token child search chunks linked via `parent_id`.

### 2. Retrieval Engines (`app/services/retrieval/`)

- [x] **Basic Vector Retriever (`basic_vector.py`)**:
  - Dense semantic search using ChromaDB and `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, runs 100% offline).
  - Cosine similarity scoring normalized to a clean `0.0` to `1.0` scale.
  - Native LangChain LCEL adapter (`as_langchain_retriever`).
- [x] **BM25 Lexical Retriever (`bm25.py`)**:
  - Inverted keyword index using `rank_bm25`.
  - Custom `LuceneBM25` implementation with $1+$ smoothing to eliminate negative or zero IDF on small document sets.
  - Handles exact keyword matches, serial numbers, error codes (e.g. `E-4021`), and acronyms.
- [x] **Hybrid Retriever with Reciprocal Rank Fusion (`hybrid.py`)**:
  - Combines ChromaDB dense vectors and BM25 sparse lexical scores.
  - Merges rankings using weighted **Reciprocal Rank Fusion (RRF $k=60$)**:
    $$\text{RRF\_Score}(d) = w_{\text{dense}} \times \frac{1}{k + \text{rank}_{\text{dense}}} + w_{\text{bm25}} \times \frac{1}{k + \text{rank}_{\text{bm25}}}$$
  - Full score transparency for the frontend Chunk Inspector (`dense_rank`, `bm25_rank`, `dense_score`, `bm25_score`, `rrf_score`).
- [x] **Cross-Encoder Re-Ranker (`reranker.py`)**:
  - Two-stage retrieval: pulls top-20 broad candidates from Hybrid search and uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score `(query, passage)` pairs simultaneously with cross-attention.
  - Distinguishes true answers from keyword-stuffed distractors.
  - Calibrated sigmoid score normalization ($0.0$ to $1.0$) and tracks `pre_rerank_rank`.
- [x] **Parent-Child Context Expansion Retriever (`parent_child.py`)**:
  - "Search small, read big": searches 200-token child chunks for sharp vector similarity, resolves matching `parent_id`, and sends the full 1000-token parent chunk to the LLM.
  - Automatic deduplication so multiple matching child chunks belonging to the same parent only return that parent once.

---

## 🧪 Automated Test Suite (43 Tests)

All modules are backed by comprehensive unit tests covering edge cases, empty strings, overlap boundaries, metadata filtering, and LCEL integration:

```bash
# Set PYTHONPATH and run all tests
$env:PYTHONPATH = "backend"
pytest backend/tests/ -v
```

### Test Summary:
```text
backend/tests/test_hybrid_retriever.py        7 PASSED
backend/tests/test_parent_child_chunking.py   5 PASSED
backend/tests/test_parent_child_retriever.py  7 PASSED
backend/tests/test_recursive_chunking.py      6 PASSED
backend/tests/test_reranker.py                6 PASSED
backend/tests/test_section_chunking.py        6 PASSED
backend/tests/test_vector_retriever.py        6 PASSED
============================= 43 passed in 31.59s =============================
```

---

## 🗺️ Next Milestones

1. **Document Ingestion Engine** (`ingestion.py`): Multi-format parser for PDF (`pypdf`), Word (`python-docx`), TXT, and Markdown.
2. **Pipeline Tracer & RAG Chain** (`pipeline_tracer.py`, `rag_chain.py`): Step latency measurement and LCEL response generation.
3. **FastAPI Endpoints** (`app/api/`): `/documents/upload`, `/query`, and `/chunks`.
4. **Query Transformations**: Multi-Query generation & HyDE (Hypothetical Document Embeddings).
5. **Next.js Frontend**: Interactive chat, Chunk Inspector, Retrieval Strategy Dropdown, and Pipeline Visualizer.
6. **Quantitative Evaluation Pipeline**: Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
