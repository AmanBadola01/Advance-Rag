from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    chunk_id: str
    chunk_index: int
    strategy: str  # "recursive", "semantic", "parent_child", "section"
    source: str
    page: Optional[int] = 1
    file_type: Optional[str] = None
    token_count: int
    char_count: int
    section: Optional[str] = None
    parent_id: Optional[str] = None
    is_parent: Optional[bool] = False
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata

class ParentChildResult(BaseModel):
    parents: list[ProcessedChunk]
    children: list[ProcessedChunk]

class RetrievalFilter(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None
    section: Optional[str] = None
    strategy: Optional[str] = None
    tags: Optional[list[str]] = None

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int
    strategy: str
    source: str
    page: Optional[int] = 1
    section: Optional[str] = None
    parent_id: Optional[str] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

