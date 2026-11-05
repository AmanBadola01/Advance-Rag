"""
Document Ingestion & Management API Endpoints.
Provides REST endpoints for file upload, strategy selection, document listing, and document deletion.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.models.schemas import IngestionSummary
from app.core.container import container
from app.services.ingestion import extract_document_pages

router = APIRouter()


# Handles multi-format document uploads, extracts text, runs the selected chunker, and indexes chunks.
@router.post("/upload", response_model=IngestionSummary, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # 1. Parse pages to store in raw documents store for future A/B on-demand chunking
        pages = extract_document_pages(file.filename, file_bytes)

        # 2. Ingest, chunk, and index into ChromaDB and BM25
        summary = container.ingestion_service.ingest_file(
            filename=file.filename,
            file_bytes=file_bytes,
            strategy=strategy,
        )

        # 3. Register raw document text for A/B testing re-chunking
        container.register_raw_document(
            doc_id=summary.doc_id,
            filename=file.filename,
            pages=pages,
            strategy=strategy,
        )

        return summary
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


# Returns the catalog of all currently ingested documents and their chunking statistics.
@router.get("", response_model=List[Dict[str, Any]])
def list_documents():
    docs = []
    for doc_id, data in container.raw_documents_store.items():
        filename = data.get("filename", "unknown")
        pages = data.get("pages", [])
        strategy = data.get("strategy", "recursive")
        docs.append({
            "doc_id": doc_id,
            "filename": filename,
            "total_pages": len(pages),
            "strategy": strategy,
        })
    return docs


# Deletes an ingested document and cascades deletion across ChromaDB, BM25, and raw stores.
@router.delete("/{source}", status_code=status.HTTP_200_OK)
def delete_document(source: str):
    if not source:
        raise HTTPException(status_code=400, detail="Source filename must be provided.")

    container.delete_document(source)
    return {"message": f"Document '{source}' and its chunks have been successfully deleted."}
