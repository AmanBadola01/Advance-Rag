"""
Chunk Inspector API Endpoints.
Serves stored document chunks, metadata, boundary diagnostics, and token counts for the frontend visualizer.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query

from app.core.container import container

router = APIRouter()


# Retrieves indexed chunks filtered by source document or chunking strategy for visual inspection in the UI.
@router.get("", response_model=List[Dict[str, Any]])
def inspect_chunks(
    source: Optional[str] = Query(None, description="Filter chunks by document source filename"),
    strategy: Optional[str] = Query(None, description="Filter chunks by chunking strategy"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of chunks to return"),
):
    all_chunks = []

    # Pull chunks from the BM25 chunk list (which holds full ProcessedChunk objects)
    for c in container.bm25_retriever.chunks:
        if source and c.metadata.source != source:
            continue
        if strategy and c.metadata.strategy != strategy:
            continue

        all_chunks.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "token_count": c.metadata.token_count,
            "char_count": c.metadata.char_count,
            "strategy": c.metadata.strategy,
            "source": c.metadata.source,
            "page": c.metadata.page,
            "section": c.metadata.section,
            "parent_id": c.metadata.parent_id,
            "is_parent": c.metadata.is_parent,
        })

        if len(all_chunks) >= limit:
            break

    # If BM25 has no chunks yet, check parent-child store
    if not all_chunks and container.parent_child_retriever.parent_store:
        for pid, p in container.parent_child_retriever.parent_store.items():
            if source and p.metadata.source != source:
                continue
            all_chunks.append({
                "chunk_id": p.chunk_id,
                "text": p.text,
                "token_count": p.metadata.token_count,
                "char_count": p.metadata.char_count,
                "strategy": "parent_child_parent",
                "source": p.metadata.source,
                "page": p.metadata.page,
                "section": p.metadata.section,
                "parent_id": None,
                "is_parent": True,
            })
            if len(all_chunks) >= limit:
                break

    return all_chunks

