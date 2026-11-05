"""
Query & A/B Comparison API Endpoints.
Executes RAG retrieval pipelines, handles on-demand custom A/B testing, and serves execution traces.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status

from app.models.schemas import QueryRequest, RAGResponse, ABTestRequest, PipelineTrace
from app.core.container import container
from app.services.pipeline_tracer import pipeline_tracer
from app.services.chunking.recursive import RecursiveChunker
from app.services.chunking.section import SectionBasedChunker
from app.services.chunking.parent_child import ParentChildChunker

router = APIRouter()


# Executes a RAG query with the selected retrieval strategy, prompt assembly, and citation tracing.
@router.post("", response_model=RAGResponse, status_code=status.HTTP_200_OK)
def execute_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        response = container.rag_chain_service.execute(
            query=req.query,
            retrieval_strategy=req.retrieval_strategy or "hybrid_rerank",
            top_k=req.top_k or 5,
            filters=req.filters,
            semantic_weight=req.semantic_weight,
            bm25_weight=req.bm25_weight,
            llm_model=req.llm_model,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


# Helper function that performs dynamic on-the-fly chunking on raw document text if custom chunk settings are provided.
def _prepare_custom_chunks_if_needed(document_id: Optional[str], config: Any) -> Optional[str]:
    if not document_id:
        return None

    raw_doc = container.get_raw_document(document_id)
    if not raw_doc:
        return None

    pages = raw_doc.get("pages", [])
    if not pages:
        return None

    strategy = (config.chunking_strategy or "recursive").lower()
    custom_strategy_tag = f"custom_{strategy}_{config.chunk_size or 500}"

    # Check if custom chunker is needed
    if strategy == "section":
        chunker = SectionBasedChunker(max_section_tokens=config.chunk_size or 800)
        chunks = chunker.split_documents(pages, doc_id=document_id, source=raw_doc["filename"])
        for c in chunks:
            c.metadata.strategy = custom_strategy_tag
        container.hybrid_retriever.add_chunks(chunks)
    elif strategy == "parent_child":
        full_text = "\n\n".join(p["text"] for p in pages)
        pc_chunker = ParentChildChunker(
            parent_chunk_size=config.parent_chunk_size or 1000,
            child_chunk_size=config.child_chunk_size or 200,
        )
        pc_result = pc_chunker.split_into_parent_child(full_text, doc_id=document_id, source=raw_doc["filename"])
        for c in pc_result.children:
            c.metadata.strategy = custom_strategy_tag
        container.parent_child_retriever.add_parent_child_result(pc_result)
    else:
        rec_chunker = RecursiveChunker(
            chunk_size=config.chunk_size or 500,
            chunk_overlap=config.chunk_overlap or 50,
        )
        chunks = rec_chunker.split_documents(pages, doc_id=document_id, source=raw_doc["filename"])
        for c in chunks:
            c.metadata.strategy = custom_strategy_tag
        container.hybrid_retriever.add_chunks(chunks)

    return custom_strategy_tag


# Executes an A/B comparison across two custom pipelines with side-by-side answers, latencies, and metrics.
@router.post("/ab-test", status_code=status.HTTP_200_OK)
def execute_ab_test(req: ABTestRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        # Dynamically prepare custom chunking if specific document ID provided
        tag_a = _prepare_custom_chunks_if_needed(req.document_id, req.pipeline_a)
        tag_b = _prepare_custom_chunks_if_needed(req.document_id, req.pipeline_b)

        from app.models.schemas import RetrievalFilter
        filter_a = RetrievalFilter(strategy=tag_a) if tag_a else None
        filter_b = RetrievalFilter(strategy=tag_b) if tag_b else None

        # Execute Pipeline A
        resp_a = container.rag_chain_service.execute(
            query=req.query,
            retrieval_strategy=req.pipeline_a.retrieval_strategy or "vector",
            top_k=req.pipeline_a.top_k or 5,
            filters=filter_a,
            semantic_weight=req.pipeline_a.semantic_weight or 0.5,
            bm25_weight=req.pipeline_a.bm25_weight or 0.5,
        )

        # Execute Pipeline B
        resp_b = container.rag_chain_service.execute(
            query=req.query,
            retrieval_strategy=req.pipeline_b.retrieval_strategy or "hybrid_rerank",
            top_k=req.pipeline_b.top_k or 5,
            filters=filter_b,
            semantic_weight=req.pipeline_b.semantic_weight or 0.5,
            bm25_weight=req.pipeline_b.bm25_weight or 0.5,
        )

        latency_delta = round(resp_a.total_latency_ms - resp_b.total_latency_ms, 2)
        faster_pipeline = "Pipeline B" if latency_delta > 0 else "Pipeline A"

        return {
            "query": req.query,
            "pipeline_a": {
                "name": req.pipeline_a.name or "Pipeline A",
                "config": req.pipeline_a.model_dump(),
                "response": resp_a,
                "latency_ms": resp_a.total_latency_ms,
            },
            "pipeline_b": {
                "name": req.pipeline_b.name or "Pipeline B",
                "config": req.pipeline_b.model_dump(),
                "response": resp_b,
                "latency_ms": resp_b.total_latency_ms,
            },
            "latency_delta_ms": abs(latency_delta),
            "faster_pipeline": faster_pipeline,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"A/B test execution failed: {str(e)}")


# Returns the step-by-step execution trace details for a given trace identifier.
@router.get("/trace/{trace_id}", response_model=PipelineTrace)
def get_execution_trace(trace_id: str):
    trace = pipeline_tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return trace


# Returns the list of recent execution traces to populate the pipeline timeline history.
@router.get("/traces", response_model=List[PipelineTrace])
def list_execution_traces(limit: int = 20):
    return pipeline_tracer.list_traces(limit=limit)

