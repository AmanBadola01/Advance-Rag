"""
Query & A/B Comparison API Endpoints.
Executes RAG retrieval pipelines, handles on-demand custom A/B testing, and serves execution traces.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Query
from contextlib import contextmanager
import uuid
import chromadb
from app.services.ingestion import normalize_pages
from app.services.retrieval.basic_vector import BasicVectorRetriever
from app.services.retrieval.bm25 import BM25Retriever
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.parent_child import ParentChildRetriever
from app.services.retrieval.reranker import HybridRerankRetriever
from app.services.rag_chain import RAGChainService

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
            retrieval_strategy=req.retrieval_strategy,
            top_k=req.top_k,
            filters=req.filters,
            semantic_weight=req.semantic_weight,
            bm25_weight=req.bm25_weight,
            llm_model=req.llm_model,
        )
        pipeline_tracer.save_answer(response.trace_id, {**response.model_dump(mode="json"), "query_scope": req.filters.source if req.filters else ""})
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"Query execution failed: {str(e)}", "trace_id": getattr(e, "trace_id", None)})


@contextmanager
def custom_pipeline(document_id, config):
    if document_id is None:
        yield container.rag_chain_service
        return
    raw_doc = container.get_raw_document(document_id)
    if raw_doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    # Each variant gets a disposable, private collection and independent lexical/parent stores.
    client = chromadb.EphemeralClient()
    name = "ab_" + uuid.uuid4().hex
    vector = BasicVectorRetriever(client=client, collection_name=name, embedding_function=container.embedding_fn)
    hybrid = HybridRetriever(vector, BM25Retriever())
    parents = ParentChildRetriever(hybrid)
    try:
        pages = normalize_pages(raw_doc["pages"], raw_doc["filename"])
        if config.chunking_strategy == "parent_child":
            chunker = ParentChildChunker(parent_chunk_size=config.parent_chunk_size, child_chunk_size=config.child_chunk_size)
            parents.add_parent_child_result(chunker.split_documents(pages, doc_id=name))
        else:
            chunker = (SectionBasedChunker(max_section_tokens=config.chunk_size)
                       if config.chunking_strategy == "section" else
                       RecursiveChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap))
            hybrid.add_chunks(chunker.split_documents(pages, doc_id=name))
        yield RAGChainService(
            vector_retriever=vector, bm25_retriever=hybrid.bm25_retriever,
            hybrid_retriever=hybrid,
            hybrid_rerank_retriever=HybridRerankRetriever(hybrid, container.cross_encoder),
            parent_child_retriever=parents, tracer=pipeline_tracer,
        )
    finally:
        client.delete_collection(name)


# Executes an A/B comparison across two custom pipelines with side-by-side answers, latencies, and metrics.
@router.post("/ab-test", status_code=status.HTTP_200_OK)
def execute_ab_test(req: ABTestRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        responses = []
        for config in (req.pipeline_a, req.pipeline_b):
            with custom_pipeline(req.document_id, config) as service:
                responses.append(service.execute(
                    query=req.query, retrieval_strategy=config.retrieval_strategy,
                    top_k=config.top_k, semantic_weight=config.semantic_weight,
                    bm25_weight=config.bm25_weight,
                ))
        for response in responses:
            pipeline_tracer.save_answer(response.trace_id, response.model_dump(mode="json"))
        resp_a, resp_b = responses

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": f"A/B test execution failed: {str(e)}", "trace_id": getattr(e, "trace_id", None)})


# Returns the step-by-step execution trace details for a given trace identifier.
@router.get("/trace/{trace_id}", response_model=PipelineTrace)
def get_execution_trace(trace_id: str):
    trace = pipeline_tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return trace


# Returns the list of recent execution traces to populate the pipeline timeline history.
@router.get("/traces", response_model=List[PipelineTrace])
def list_execution_traces(limit: int = Query(20, ge=1, le=100)):
    return pipeline_tracer.list_traces(limit=limit)

