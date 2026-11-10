from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, model_validator

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
    saved_answer: Optional[Dict[str, Any]] = None
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


RetrievalStrategy = Literal["vector", "bm25", "hybrid", "hybrid_rerank", "parent_child"]


class RetrievalConfig(BaseModel):
    retrieval_strategy: RetrievalStrategy = "hybrid_rerank"
    top_k: int = Field(5, ge=1, le=100)
    semantic_weight: float = Field(0.5, ge=0, le=1)
    bm25_weight: float = Field(0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_weights(self):
        if self.semantic_weight + self.bm25_weight <= 0:
            raise ValueError("At least one retrieval weight must be positive.")
        return self


class QueryRequest(RetrievalConfig):
    query: str
    filters: Optional[RetrievalFilter] = None
    llm_model: Optional[str] = None


class CustomPipelineConfig(RetrievalConfig):
    name: str = "Pipeline"
    chunking_strategy: Literal["recursive", "section", "parent_child"] = "recursive"
    chunk_size: int = Field(500, ge=1, le=10000)
    chunk_overlap: int = Field(50, ge=0)
    parent_chunk_size: int = Field(1000, ge=101, le=10000)
    child_chunk_size: int = Field(200, ge=26, le=10000)

    @model_validator(mode="after")
    def validate_chunk_sizes(self):
        if self.chunking_strategy == "recursive" and self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        if self.chunking_strategy == "parent_child" and self.child_chunk_size > self.parent_chunk_size:
            raise ValueError("Child chunks must not exceed parent chunks.")
        return self


class ABTestRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    pipeline_a: CustomPipelineConfig = Field(default_factory=lambda: CustomPipelineConfig(name="Pipeline A", retrieval_strategy="vector"))
    pipeline_b: CustomPipelineConfig = Field(default_factory=lambda: CustomPipelineConfig(name="Pipeline B", retrieval_strategy="hybrid_rerank"))

