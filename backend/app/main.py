"""
FastAPI Application Entry Point.
Configures CORS middleware for Next.js, mounts modular API routers,
and exposes health monitoring endpoints for the Advanced RAG Platform.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.setup_models import models_ready
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

from app.auth import router as auth_router, current_user
from app.core.account_context import account_id
from urllib.parse import urlsplit
app.include_router(auth_router)

@app.middleware('http')
async def account_boundary(request, call_next):
    path = request.url.path
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        origin = request.headers.get('origin')
        if origin and urlsplit(origin).netloc != request.headers.get('host'):
            return JSONResponse(status_code=403, content={'detail': 'Cross-origin request refused.'})
    protected = (path.startswith('/api/') and not path.startswith('/api/auth/')) or path in ('/health', '/workspace')
    user = current_user(request) if protected else None
    if protected and not user:
        if path == '/workspace':
            from fastapi.responses import RedirectResponse
            return RedirectResponse('/login', status_code=303)
        return JSONResponse(status_code=401, content={'detail': 'Please sign in.'})
    token = account_id.set(user['id'] if user else None)
    try:
        response = await call_next(request)
        if protected or path.startswith('/api/auth/'):
            response.headers['Cache-Control'] = 'no-store'
        return response
    finally:
        account_id.reset(token)

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

    ready = models_ready()
    consistent = (vector_count == bm25_count and
                  all(c.metadata.parent_id in container.parent_child_retriever.parent_store
                      for c in container.bm25_retriever.chunks if c.metadata.parent_id) and
                  (not vector_count or bool(container.raw_documents_store)))
    return JSONResponse(status_code=200 if ready and consistent else 503, content={
        "status": "healthy" if ready and consistent else "not_ready",
        "models_ready": ready,
        "storage_consistent": consistent,
        "setup_hint": None if ready else "Run python -m app.setup_models to provision local models.",
        "vector_chunks": vector_count,
        "bm25_chunks": bm25_count,
        "parent_chunks": parent_count,
        "raw_documents_cached": len(container.raw_documents_store),
    })

