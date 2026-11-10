"""
RAG Chain Orchestrator Service.
Coordinates dynamic retriever selection, LangChain LCEL prompt assembly,
inline citation formatting, LLM synthesis (Gemini/OpenAI with offline fallback),
and end-to-end execution tracing.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.models.schemas import (
    RetrievedChunk,
    CitationItem,
    RAGResponse,
    RetrievalFilter,
)
from app.services.pipeline_tracer import pipeline_tracer, PipelineTracer
from app.services.retrieval.basic_vector import BasicVectorRetriever
from app.services.retrieval.bm25 import BM25Retriever
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reranker import HybridRerankRetriever
from app.services.retrieval.parent_child import ParentChildRetriever


SYSTEM_RAG_PROMPT = """You are an expert AI assistant answering questions based strictly on the provided retrieved context.

Instructions:
1. Answer the question thoroughly, accurately, and directly using only the provided context snippets.
2. If the answer cannot be found in the context, explicitly state: "Based on the provided documents, I could not find sufficient information to answer this question."
3. Every time you mention a fact from a context snippet, cite its source number using brackets, for example [1] or [2].
4. Do not invent facts, citations, or speculate outside of what is explicitly documented.

Retrieved Context:
{context}
"""


# Formats retrieved chunks into indexed context strings with bracket citations and extracts citation metadata.
def format_context_with_citations(chunks: List[RetrievedChunk]) -> Tuple[str, List[CitationItem]]:
    if not chunks:
        return "No relevant context found.", []

    formatted_parts: List[str] = []
    citations: List[CitationItem] = []

    for idx, chunk in enumerate(chunks, start=1):
        source_name = chunk.source or "unknown_source"
        page_num = chunk.page or 1
        section_name = chunk.section

        header = f"[{idx}] (Source: {source_name}, Page: {page_num}"
        if section_name:
            header += f", Section: {section_name}"
        header += f", Relevance Score: {round(chunk.score, 4)})"

        body = chunk.text.strip()
        formatted_parts.append(f"{header}\n{body}")

        snippet_preview = body[:160] + ("..." if len(body) > 160 else "")
        citations.append(
            CitationItem(
                index=idx,
                chunk_id=chunk.chunk_id,
                source=source_name,
                page=page_num,
                section=section_name,
                score=round(chunk.score, 4),
                snippet=snippet_preview,
            )
        )

    context_str = "\n\n" + ("=" * 40) + "\n\n".join(formatted_parts)
    return context_str, citations


# Instantiates the active chat model (Gemini or OpenAI) based on available API keys with offline fallback.
def get_llm_instance(model_name: Optional[str] = None):
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            selected_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            return ChatGoogleGenerativeAI(
                model=selected_model,
                temperature=0.1,
                google_api_key=google_key,
                timeout=30,
                max_retries=0,
            ), selected_model
        except Exception:
            pass

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            selected_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                api_key=openai_key,
                timeout=30,
                max_retries=0,
            ), selected_model
        except Exception:
            pass

    return None, "offline-synthesizer"


# Synthesizes a structured extractive answer directly from top retrieved chunks when running completely offline.
def generate_offline_fallback_answer(query: str, chunks: List[RetrievedChunk], citations: List[CitationItem]) -> str:
    if not chunks:
        return "Based on the provided documents, I could not find sufficient information to answer this question."

    top_chunks = chunks[:3]
    answer_bullets = []

    for idx, c in enumerate(top_chunks, start=1):
        first_sentences = re.split(r"(?<=[.!?])\s+", c.text.strip())
        summary_sentence = " ".join(first_sentences[:2]).strip()
        if summary_sentence:
            answer_bullets.append(f"• {summary_sentence} [{idx}]")

    sources_summary = ", ".join(f"{c.source} (Page {c.page})" for c in citations[:3])
    
    synthesized_text = (
        f"Based on the retrieved documentation from {sources_summary}:\n\n"
        + "\n".join(answer_bullets)
        + "\n\n*(Note: Generated via local extractive synthesizer. Add GOOGLE_API_KEY or OPENAI_API_KEY in .env for generative LLM synthesis.)*"
    )
    return synthesized_text


class RAGChainService:
    """
    RAG Chain Service:
    Coordinates retriever dispatching, prompt preparation, LLM execution,
    citation linking, and step-by-step pipeline tracing.
    """

    # Initializes the service with storage retriever engines and tracing service.
    def __init__(
        self,
        vector_retriever: Optional[BasicVectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        hybrid_rerank_retriever: Optional[HybridRerankRetriever] = None,
        parent_child_retriever: Optional[ParentChildRetriever] = None,
        tracer: Optional[PipelineTracer] = None,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = hybrid_retriever
        self.hybrid_rerank_retriever = hybrid_rerank_retriever
        self.parent_child_retriever = parent_child_retriever
        self.tracer = tracer or pipeline_tracer

    # Dispatches the requested retrieval strategy key to the corresponding retriever engine.
    def _resolve_retriever(self, strategy: str):
        strat = strategy.lower().strip()
        if strat in ("hybrid_rerank", "rerank", "hybrid_reranker"):
            return self.hybrid_rerank_retriever or self.hybrid_retriever or self.vector_retriever
        elif strat in ("hybrid", "rrf"):
            return self.hybrid_retriever or self.vector_retriever
        elif strat in ("parent_child", "parent_child_expansion"):
            return self.parent_child_retriever or self.vector_retriever
        elif strat in ("bm25", "lexical", "keyword"):
            return self.bm25_retriever
        elif strat in ("vector", "basic_vector", "dense"):
            return self.vector_retriever
        else:
            return self.hybrid_rerank_retriever or self.hybrid_retriever or self.vector_retriever

    # Executes the full RAG pipeline: retrieval, context assembly, LLM synthesis, and execution tracing.
    def execute(
        self,
        query: str,
        retrieval_strategy: str = "hybrid_rerank",
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
        semantic_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        llm_model: Optional[str] = None,
    ) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        # 1. Initialize Pipeline Trace
        trace_id = self.tracer.start_trace(query=query, retrieval_strategy=retrieval_strategy)

        try:
            # 2. Execute Retrieval Step
            retriever = self._resolve_retriever(retrieval_strategy)
            retrieved_chunks: List[RetrievedChunk] = []

            with self.tracer.step_timer(trace_id, "Candidate Retrieval", {"strategy": retrieval_strategy, "top_k": top_k}) as step_meta:
                if retriever:
                    # Handle strategy-specific parameter passes
                    if hasattr(retriever, "retrieve"):
                        try:
                            retrieved_chunks = retriever.retrieve(
                                query=query,
                                top_k=top_k,
                                filters=filters,
                                semantic_weight=semantic_weight,
                                bm25_weight=bm25_weight,
                            )
                        except TypeError:
                            # Fallback for retrievers without weight kwargs (vector, bm25, parent_child)
                            retrieved_chunks = retriever.retrieve(
                                query=query,
                                top_k=top_k,
                                filters=filters,
                            )
                    elif hasattr(retriever, "invoke"):
                        lc_docs = retriever.invoke(query)
                        retrieved_chunks = [
                            RetrievedChunk(
                                chunk_id=doc.metadata.get("chunk_id", f"c_{i}"),
                                text=doc.page_content,
                                score=float(doc.metadata.get("score", 1.0)),
                                rank=i + 1,
                                strategy=retrieval_strategy,
                                source=doc.metadata.get("source", "unknown"),
                                page=doc.metadata.get("page", 1),
                            )
                            for i, doc in enumerate(lc_docs)
                        ]

                step_meta["chunks_retrieved"] = len(retrieved_chunks)

            # 3. Context & Citation Assembly Step
            with self.tracer.step_timer(trace_id, "Context Assembly") as step_meta:
                context_text, citations = format_context_with_citations(retrieved_chunks)
                step_meta["citations_count"] = len(citations)
                step_meta["context_char_length"] = len(context_text)

            # 4. LLM Generation Step
            final_answer = ""
            llm, model_identifier = get_llm_instance(model_name=llm_model)
            estimated_tokens = len(query.split()) + len(context_text.split())

            with self.tracer.step_timer(trace_id, "LLM Synthesis", {"model": model_identifier}) as step_meta:
                if llm is not None:
                    try:
                        # LangChain LCEL prompt chain
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", SYSTEM_RAG_PROMPT),
                            ("user", "{question}"),
                        ])
                        chain = prompt | llm | StrOutputParser()
                        final_answer = chain.invoke({"context": context_text, "question": query})
                    except Exception as e:
                        # Graceful fallback if external LLM API rate limits or errors
                        final_answer = generate_offline_fallback_answer(query, retrieved_chunks, citations)
                        step_meta["api_fallback"] = True
                        step_meta["error"] = str(e)
                else:
                    final_answer = generate_offline_fallback_answer(query, retrieved_chunks, citations)

                estimated_tokens += len(final_answer.split())
                step_meta["output_length"] = len(final_answer)

            # 5. Finalize Execution Trace
            completed_trace = self.tracer.end_trace(trace_id=trace_id, total_tokens=estimated_tokens)

            return RAGResponse(
                query=query,
                answer=final_answer,
                retrieval_strategy=retrieval_strategy,
                retrieved_chunks=retrieved_chunks,
                citations=citations,
                trace_id=trace_id,
                total_latency_ms=completed_trace.total_latency_ms,
                model_name=model_identifier,
            )

        except Exception as exc:
            if trace_id in self.tracer._active_contexts:
                self.tracer.end_trace(trace_id, total_tokens=0)
            exc.trace_id = trace_id
            raise

    # Runs two independent pipeline configurations concurrently on the same query for side-by-side A/B evaluation.
    def execute_ab_comparison(
        self,
        query: str,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Execute Pipeline A
        resp_a = self.execute(
            query=query,
            retrieval_strategy=config_a.get("retrieval_strategy", "vector"),
            top_k=config_a.get("top_k", 5),
            semantic_weight=config_a.get("semantic_weight", 0.5),
            bm25_weight=config_a.get("bm25_weight", 0.5),
        )

        # Execute Pipeline B
        resp_b = self.execute(
            query=query,
            retrieval_strategy=config_b.get("retrieval_strategy", "hybrid_rerank"),
            top_k=config_b.get("top_k", 5),
            semantic_weight=config_b.get("semantic_weight", 0.5),
            bm25_weight=config_b.get("bm25_weight", 0.5),
        )

        return {
            "query": query,
            "pipeline_a": {
                "config": config_a,
                "response": resp_a,
                "latency_ms": resp_a.total_latency_ms,
            },
            "pipeline_b": {
                "config": config_b,
                "response": resp_b,
                "latency_ms": resp_b.total_latency_ms,
            },
            "latency_delta_ms": round(resp_a.total_latency_ms - resp_b.total_latency_ms, 2),
        }

