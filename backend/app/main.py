"""
FastAPI Application Entry Point.
Configures CORS middleware for Next.js, mounts modular API routers,
and exposes health monitoring endpoints for the Advanced RAG Platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.chunks import router as chunks_router
from app.core.container import container

app = FastAPI(
    title="Advanced RAG Platform API",
    description="Production-grade RAG backend with Hybrid Search, Cross-Encoder Re-Ranking, Parent-Child Context Expansion, and A/B Testing.",
    version="1.0.0",
)

# Configure Cross-Origin Resource Sharing (CORS) for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])
app.include_router(query_router, prefix="/api/query", tags=["Query & A/B Testing"])
app.include_router(chunks_router, prefix="/api/chunks", tags=["Chunk Inspector"])


# Returns service metadata, supported retrieval strategies, and system status.
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "Advanced RAG Platform Backend",
        "version": "1.0.0",
        "supported_strategies": {
            "chunking": ["recursive", "section", "parent_child"],
            "retrieval": ["vector", "bm25", "hybrid", "hybrid_rerank", "parent_child"],
        },
        "docs_url": "/docs",
    }


# Health check endpoint returning database connectivity and indexed chunk counts.
@app.get("/health", tags=["Health"])
def health_check():
    vector_count = container.vector_retriever.count()
    bm25_count = container.bm25_retriever.count()
    parent_count = len(container.parent_child_retriever.parent_store)

    return {
        "status": "healthy",
        "vector_chunks": vector_count,
        "bm25_chunks": bm25_count,
        "parent_chunks": parent_count,
        "raw_documents_cached": len(container.raw_documents_store),
    }

