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

