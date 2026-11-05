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


class IngestionSummary(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    strategy: str
    chunk_count: int
    total_tokens: int
    created_at: str


class TraceStep(BaseModel):
    name: str
    duration_ms: float
    status: str = "success"
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class PipelineTrace(BaseModel):
    trace_id: str
    query: str
    retrieval_strategy: str
    steps: list[TraceStep]
    total_latency_ms: float
    total_tokens: int = 0
    created_at: str


class CitationItem(BaseModel):
    index: int
    chunk_id: str
    source: str
    page: Optional[int] = 1
    section: Optional[str] = None
    score: float
    snippet: str


class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieval_strategy: str
    retrieved_chunks: list[RetrievedChunk]
    citations: list[CitationItem]
    trace_id: str
    total_latency_ms: float
    model_name: str


class QueryRequest(BaseModel):
    query: str
    retrieval_strategy: Optional[str] = "hybrid_rerank"
    top_k: Optional[int] = 5
    filters: Optional[RetrievalFilter] = None
    semantic_weight: Optional[float] = 0.5
    bm25_weight: Optional[float] = 0.5
    llm_model: Optional[str] = None


class CustomPipelineConfig(BaseModel):
    name: Optional[str] = "Pipeline"
    chunking_strategy: Optional[str] = "recursive"
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50
    parent_chunk_size: Optional[int] = 1000
    child_chunk_size: Optional[int] = 200
    retrieval_strategy: Optional[str] = "hybrid_rerank"
    top_k: Optional[int] = 5
    semantic_weight: Optional[float] = 0.5
    bm25_weight: Optional[float] = 0.5


class ABTestRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    pipeline_a: CustomPipelineConfig = Field(default_factory=lambda: CustomPipelineConfig(name="Pipeline A", retrieval_strategy="vector"))
    pipeline_b: CustomPipelineConfig = Field(default_factory=lambda: CustomPipelineConfig(name="Pipeline B", retrieval_strategy="hybrid_rerank"))

