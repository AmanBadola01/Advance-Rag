"""API and lifecycle regression coverage for the QA findings.

ML substitutes keep these tests fast; the existing retriever suite tests real models.
"""
import importlib
import io
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    from app.services.retrieval.base import LocalMiniLMEmbedding
    from app.services.retrieval.reranker import CrossEncoderReranker
    class Embeddings:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, float("quartz" in t.lower()), 0.5] for t in texts])
    class Reranker:
        def predict(self, pairs):
            return np.array([1.0 for _ in pairs])
    monkeypatch.setattr(LocalMiniLMEmbedding, "_get_model", lambda self: Embeddings())
    monkeypatch.setattr(CrossEncoderReranker, "_get_model", lambda self: Reranker())
    from app.core.container import ServiceContainer
    service = ServiceContainer(str(tmp_path / "chroma"))
    for name in ("app.main", "app.api.documents", "app.api.query", "app.api.chunks"):
        monkeypatch.setattr(importlib.import_module(name), "container", service)
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "models_ready", lambda: True)
    from app.services.pipeline_tracer import pipeline_tracer
    pipeline_tracer.clear()
    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client, service


def upload(client, strategy="recursive", name="quartz.txt", data=None):
    return client.post("/api/documents/upload", files={"file": (
        name, data or b"Quartz launch code is ALPHA-729. FastAPI powers Quartz."
    )}, data={"strategy": strategy})


@pytest.mark.parametrize("chunking", ["recursive", "section", "parent_child"])
def test_upload_search_trace_delete(api, chunking):
    client, service = api
    response = upload(client, chunking)
    assert response.status_code == 201, response.text
    assert client.get("/api/documents").json()[0]["filename"] == "quartz.txt"
    for retrieval in ("vector", "bm25", "hybrid", "hybrid_rerank", "parent_child"):
        result = client.post("/api/query", json={"query": "Quartz", "retrieval_strategy": retrieval})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["retrieved_chunks"] and body["citations"]
        assert all(c["source"] == "quartz.txt" for c in body["retrieved_chunks"])
        assert client.get("/api/query/trace/" + body["trace_id"]).status_code == 200
    assert client.delete("/api/documents/quartz.txt").status_code == 200
    assert service.vector_retriever.count() == service.bm25_retriever.count() == 0
    assert not service.parent_child_retriever.parent_store


@pytest.mark.parametrize("strategy", ["recursive", "section", "parent_child"])
def test_document_scoped_ab_isolation(api, strategy):
    client, service = api
    did = upload(client).json()["doc_id"]
    before = service.vector_retriever.collection.get()
    config = {"chunking_strategy": strategy, "retrieval_strategy": "bm25", "chunk_overlap": 0,
              "semantic_weight": 0, "bm25_weight": 1}
    for _ in range(2):
        result = client.post("/api/query/ab-test", json={"query": "Quartz", "document_id": did,
                             "pipeline_a": config, "pipeline_b": {**config, "chunk_size": 300,
                             "parent_chunk_size": 600}})
        assert result.status_code == 200, result.text
        body = result.json()
        a = body["pipeline_a"]["response"]["retrieved_chunks"]
        b = body["pipeline_b"]["response"]["retrieved_chunks"]
        assert a and b and a[0]["chunk_id"] != b[0]["chunk_id"]
        assert service.vector_retriever.collection.get() == before
        assert service.bm25_retriever.count() == len(before["ids"])
    assert client.post("/api/query/ab-test", json={"query": "Quartz", "document_id": "missing"}).status_code == 404


def test_restart_restores_every_store_and_delete(api):
    client, service = api
    upload(client, "parent_child")
    from app.core.container import ServiceContainer
    restored = ServiceContainer(service.db_path)
    assert restored.raw_documents_store
    assert restored.parent_child_retriever.parent_store
    assert restored.vector_retriever.count() == restored.bm25_retriever.count() > 0
    assert restored.bm25_retriever.retrieve("Quartz")
    # A fresh interpreter proves restoration does not rely on Python singletons.
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    code = "from app.core.container import container as c; assert c.raw_documents_store; assert c.parent_child_retriever.parent_store; assert c.bm25_retriever.retrieve('Quartz')"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    restored.delete_document("quartz.txt")
    final = ServiceContainer(service.db_path)
    assert not final.raw_documents_store and not final.parent_child_retriever.parent_store
    assert final.bm25_retriever.count() == final.vector_retriever.count() == 0


@pytest.mark.parametrize("extra", [
    {"top_k": -1}, {"top_k": 0}, {"top_k": 101}, {"top_k": None},
    {"semantic_weight": -2}, {"semantic_weight": 0, "bm25_weight": 0},
    {"retrieval_strategy": "invalid"},
])
def test_query_validation(api, extra):
    assert api[0].post("/api/query", json={"query": "Quartz", **extra}).status_code == 422


def test_invalid_files_and_chunk_configs(api):
    client, _ = api
    assert upload(client, name="bad.pdf", data=b"not a PDF").status_code == 400
    assert upload(client, name="bad.docx", data=b"not a ZIP").status_code == 400
    assert upload(client, name="bad.exe").status_code == 400
    assert upload(client, strategy="typo").status_code == 400
    for config in ({"chunk_size": 20, "chunk_overlap": 20},
                   {"chunking_strategy": "parent_child", "child_chunk_size": 2000}):
        assert client.post("/api/query/ab-test", json={"query": "Quartz", "pipeline_a": config}).status_code == 422


def test_inspector_filters_and_failed_traces(api):
    client, service = api
    upload(client, "parent_child")
    assert client.get("/api/chunks?strategy=nonexistent").json() == []
    assert client.get("/api/chunks?strategy=parent_child_parent").json()[0]["is_parent"]
    with patch.object(service.vector_retriever, "retrieve", side_effect=RuntimeError("forced storage failure")):
        result = client.post("/api/query", json={"query": "Quartz", "retrieval_strategy": "vector"})
    assert result.status_code == 500
    trace_id = result.json()["detail"]["trace_id"]
    trace = client.get("/api/query/trace/" + trace_id).json()
    assert trace["steps"][0]["status"] == "failed"
    from app.services.pipeline_tracer import pipeline_tracer
    assert not pipeline_tracer._active_contexts


def test_health_missing_models(api, monkeypatch):
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "models_ready", lambda: False)
    result = api[0].get("/health")
    assert result.status_code == 503 and result.json()["setup_hint"]


def test_page_metadata_and_docx(api):
    from app.services.ingestion import normalize_pages
    from app.services.chunking.section import SectionBasedChunker
    pages = normalize_pages([{"text": "# First\nQuartz", "page": 1}, {"text": "# Second\nQuartz", "page": 2}], "source.pdf")
    chunks = SectionBasedChunker().split_documents(pages, doc_id="pages")
    assert [c.metadata.page for c in chunks] == [1, 2]
    assert len({c.chunk_id for c in chunks}) == 2
    assert all(c.chunk_id == c.metadata.chunk_id and c.metadata.source == "source.pdf" for c in chunks)
    import docx
    doc = docx.Document(); doc.add_paragraph("Quartz launch code is ALPHA-729.")
    data = io.BytesIO(); doc.save(data)
    assert upload(api[0], name="source.docx", data=data.getvalue()).status_code == 201


def test_valid_pdf_upload(api):
    # A text-bearing PDF fixture, built without adding a PDF generation dependency.
    content = b"BT /F1 12 Tf 50 700 Td (Quartz launch code ALPHA-729.) Tj ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"]
    pdf = b"%PDF-1.4\n"; offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf)); pdf += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 6\n0000000000 65535 f \n"
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    pdf += f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    client, _ = api
    assert upload(client, name="source.pdf", data=pdf).status_code == 201
    result = client.post("/api/query", json={"query": "Quartz", "retrieval_strategy": "bm25"}).json()
    assert result["citations"][0]["source"] == "source.pdf"
    assert "ALPHA-729" in result["answer"]
