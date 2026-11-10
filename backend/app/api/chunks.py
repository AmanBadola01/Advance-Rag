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
    chunks = list(container.bm25_retriever.chunks) + list(container.parent_child_retriever.parent_store.values())
    result = []
    seen = set()
    for chunk in chunks:
        meta = chunk.metadata
        label = "parent_child_parent" if meta.is_parent else meta.strategy
        if source and meta.source != source:
            continue
        if strategy and strategy != label:
            continue
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        result.append({
            "chunk_id": chunk.chunk_id, "text": chunk.text,
            "token_count": meta.token_count, "char_count": meta.char_count,
            "strategy": label, "source": meta.source, "page": meta.page,
            "section": meta.section, "parent_id": meta.parent_id,
            "is_parent": meta.is_parent,
        })
        if len(result) >= limit:
            break
    return result
