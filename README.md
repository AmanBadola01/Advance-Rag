# Retrieval Studio

An interactive Retrieval-Augmented Generation (RAG) workspace for exploring documents, comparing retrieval strategies, and producing answers with traceable source citations.

**[Open the live application](http://116.202.210.102:20304/login)** · [Repository](https://github.com/AmanBadola01/Advance-Rag)

Create an account or sign in to use the workspace. Each account has its own documents, indexes, and saved query history. The current deployment uses HTTP on the address above.

## Features

- **Document management:** upload PDF, DOCX, TXT, Markdown, CSV, and JSON files; explicitly chunk and index them; inspect the catalog and delete documents.
- **Three chunking strategies:** recursive splitting, section-based splitting, and hierarchical parent–child splitting.
- **Five retrieval strategies:** semantic vector search, BM25 keyword search, hybrid search, hybrid search with cross-encoder re-ranking, and parent–child context expansion.
- **Grounded answers:** formatted Markdown, numbered citations, source passages, evidence scores, and document-level query scope.
- **Strategy comparisons:** compare two pipelines over an existing corpus or a selected document, with latency, answers, and retrieved evidence.
- **Chunk explorer:** inspect text, metadata, token counts, and parent context with filters and loaded-result search.
- **Pipeline visibility:** inspect retrieval/generation steps, durations, estimated token usage, and saved answers. Provider failures show an explicit extraction fallback.
- **Account access:** signup/login, salted scrypt password hashes, expiring HttpOnly sessions, and isolated persistent account storage.
- **Integrated guide:** in-app explanations of chunking, retrieval, workflow, and troubleshooting.

## Technology

| Layer | Implementation |
| --- | --- |
| Interface | HTML, CSS, vanilla JavaScript; locally bundled Marked and DOMPurify |
| API | Python, FastAPI, Uvicorn, Pydantic |
| RAG orchestration | LangChain / LCEL |
| Vector storage | ChromaDB |
| Embeddings | Sentence Transformers: `all-MiniLM-L6-v2` |
| Keyword retrieval | BM25 through `rank_bm25` |
| Re-ranking | Cross-encoder: `ms-marco-MiniLM-L-6-v2` |
| Generation | Gemini by default; OpenAI support in the backend |
| Deployment | Single Uvicorn worker with PM2 supervision |
| UI checks | Node.js test runner and jsdom |

The interface is served by FastAPI from the same origin as the API. No frontend build step or Node.js production server is required.

## How it works

1. Sign up or sign in, then open **Documents**.
2. Add files, select a chunking strategy, and choose **Chunk and index documents**.
3. In **Playground**, select all documents or a specific source, choose a retriever, and ask a question.
4. Review the answer, source citations, retrieved evidence, and pipeline trace.
5. Use **Compare strategies** to examine two configurations or **Chunk explorer** to inspect indexed content.

Hybrid retrieval combines dense and BM25 rankings through reciprocal rank fusion. The re-ranking strategy scores the candidate passages with a cross-encoder before generation. Parent–child retrieval searches small chunks and expands matching results to their parent context.

## Local setup

### Requirements

- Python 3.12 and pip.
- Internet access for the initial embedding/re-ranking model downloads.
- A Google API key for Gemini generation, or an OpenAI key for the supported alternative. Without a working generation provider, the application can return an explicitly labeled extractive fallback.
- Node.js and npm only if running frontend checks.

### Install and configure

```bash
git clone https://github.com/AmanBadola01/Advance-Rag.git
cd Advance-Rag
python3 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
```

Set `GOOGLE_API_KEY` in your local `.env` or inject it through the environment. Never commit credentials. Existing process environment variables take precedence over `.env` values.

Download the local retrieval models once:

```bash
PYTHONPATH=backend python -m app.setup_models
```

Start the integrated application:

```bash
PORT=8000 python run_workspace.py
```

Open [http://localhost:8000/login](http://localhost:8000/login). The authenticated interface is at `/workspace`; API documentation is at `/docs`, and readiness is at `/health`.

### Configuration

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini provider credential |
| `OPENAI_API_KEY` | Optional alternative generation credential |
| `GEMINI_MODEL` | Override the default Gemini model, currently `gemini-3.5-flash-lite`; select one available to your provider account |
| `PORT` | Integrated server port; launcher default is `20318` |
| `CHROMA_PERSIST_DIRECTORY` | Base/legacy vector persistence directory |
| `RAG_ACCOUNT_DIRECTORY` | Account records, sessions, and isolated account data; default `backend/data/accounts` |
| `HF_HOME` | Persistent Hugging Face model cache location |
| `TIKTOKEN_CACHE_DIR` | Optional persistent tokenizer cache location |
| `RAG_MODEL_LOCAL_ONLY` | Set `true` after provisioning to load retrieval models offline |

The launcher also accepts an injected `SECRET_GEMINI_API_KEY` and maps it to `GOOGLE_API_KEY` in memory. Run one worker: in-memory indexes and storage locks are process-local. Preserve and back up account storage across restarts; it is intentionally excluded from Git.

## Repository layout

```text
Advance-Rag/
├── backend/
│   ├── app/
│   │   ├── api/                 # Documents, queries, comparisons, chunks
│   │   ├── core/                # Service container and account context
│   │   ├── models/              # API data contracts
│   │   ├── services/            # Ingestion, chunking, retrieval, generation, traces
│   │   ├── auth.py             # Account and session handling
│   │   ├── integrated.py       # Same-origin frontend entry point
│   │   └── setup_models.py     # Retrieval model provisioning
│   ├── tests/                  # Backend tests
│   └── requirements.txt
├── frontend/                   # Interface, assets, bundled libraries, UI tests
├── deploy/                     # PM2 configuration and deployment notes
├── .env.example                # Placeholder configuration
└── run_workspace.py            # Integrated application launcher
```

## Verification

```bash
# Backend checks; retrieval tests require provisioned models
RAYON_NUM_THREADS=2 TOKIO_WORKER_THREADS=2 PYTHONPATH=backend \
  backend/venv/bin/python -m pytest backend/tests -q

# Frontend checks
npm --prefix frontend ci
npm --prefix frontend test
```

Backend coverage includes chunking, retrieval, API regression behavior, account isolation, persistence, and credential validation. Frontend checks cover answer rendering, account forms, document scope, uploads, refresh controls, help navigation, greetings, and validation.

## Deployment and operational limits

The current live deployment is [Retrieval Studio](http://116.202.210.102:20304/login). See [deployment notes](deploy/README.md) for the existing PM2 setup and [integration notes](frontend/INTEGRATION.md) for implementation details.

- HTTPS, a custom domain, email verification, and password recovery are not configured.
- PM2 restarts the application process; full-server reboot recovery is not configured.
- Model caches and account data must remain on persistent storage. Do not commit them or serialize injected credentials with `pm2 save`.
- Scanned PDFs require external OCR; the application does not provide OCR.
- Same-name document uploads share filename-based filtering/deletion behavior.
- Chunk inspection returns at most 200 rows; text search applies to the loaded subset.
- Trace token counts are estimates, and comparison latency excludes index preparation.
- Automated quality evaluation, HyDE, and multi-query rewriting are not implemented.

The live IP/port remains dependent on the hosting allocation. The repository contains the application source and deployment configuration, not private uploaded documents, account databases, credentials, or downloaded model artifacts.
