"""
Document Ingestion Service.
Parses multi-format files (PDF, DOCX, TXT, Markdown), extracts page numbers and text,
executes the selected chunking strategy, and indexes chunks into vector and lexical stores.
"""

import io
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union

import pypdf
import docx

from app.models.schemas import IngestionSummary, ProcessedChunk, ParentChildResult
from app.services.chunking.recursive import RecursiveChunker
from app.services.chunking.section import SectionBasedChunker
from app.services.chunking.parent_child import ParentChildChunker


# Sanitizes raw text by removing null bytes, normalizing line endings, and collapsing redundant blank lines.
def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Extracts text from PDF bytes page-by-page while tracking individual page numbers.
def extract_text_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_extracted_text(raw_text)
        if cleaned:
            pages.append({"text": cleaned, "page": page_idx + 1})

    return pages


# Extracts paragraph and table content from Microsoft Word DOCX bytes.
def extract_text_from_docx(file_bytes: bytes) -> List[Dict[str, Any]]:
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs: List[str] = []

    for p in doc.paragraphs:
        if p.text and p.text.strip():
            paragraphs.append(p.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    full_text = clean_extracted_text("\n\n".join(paragraphs))
    if not full_text:
        return []

    return [{"text": full_text, "page": 1}]


# Decodes plain text or Markdown bytes into Unicode strings with automatic fallback decoding.
def extract_text_from_txt(file_bytes: bytes) -> List[Dict[str, Any]]:
    try:
        decoded = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = file_bytes.decode("latin-1", errors="ignore")

    cleaned = clean_extracted_text(decoded)
    if not cleaned:
        return []

    return [{"text": cleaned, "page": 1}]


# Resolves the normalized lowercase file extension from a filename.
def detect_file_type(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext.lstrip(".")


# Dispatches byte content to the appropriate format extractor based on file extension.
def extract_document_pages(filename: str, file_bytes: bytes) -> List[Dict[str, Any]]:
    if not file_bytes:
        return []

    file_type = detect_file_type(filename)

    if file_type not in {"pdf", "docx", "txt", "md", "markdown", "csv", "json"}:
        raise ValueError(f"Unsupported file type: {file_type or 'unknown'}")
    try:
        if file_type == "pdf":
            return extract_text_from_pdf(file_bytes)
        if file_type == "docx":
            return extract_text_from_docx(file_bytes)
        return extract_text_from_txt(file_bytes)
    except Exception as exc:
        raise ValueError(f"Cannot read {file_type} document: {exc}") from exc


def normalize_pages(pages, filename, custom_metadata=None):
    return [{"text": p["text"], "metadata": {
        **(custom_metadata or {}), "source": filename,
        "page": p.get("page", 1), "file_type": detect_file_type(filename),
    }} for p in pages]



class DocumentIngestionService:
    """
    Document Ingestion Pipeline:
    1. Extracts text and page structures from uploaded documents.
    2. Applies the selected chunking strategy (recursive, section, parent_child).
    3. Indexes generated chunks into vector (ChromaDB) and sparse (BM25) search engines.
    4. Produces a detailed IngestionSummary for the frontend dashboard.
    """

    # Initializes the ingestion service with storage retrievers and chunking engine instances.
    def __init__(
        self,
        hybrid_retriever: Optional[Any] = None,
        parent_child_retriever: Optional[Any] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.parent_child_retriever = parent_child_retriever

        # Pre-instantiate reusable chunkers
        self.recursive_chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        self.section_chunker = SectionBasedChunker(max_section_tokens=800)
        self.parent_child_chunker = ParentChildChunker(
            parent_chunk_size=1000,
            parent_chunk_overlap=100,
            child_chunk_size=200,
            child_chunk_overlap=25,
        )

    # Ingests, chunks and indexes an uploaded file into search engines according to the chosen strategy.
    def ingest_file(
        self,
        filename: str,
        file_bytes: bytes,
        strategy: str = "recursive",
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionSummary:
        if not file_bytes:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        if strategy not in {"recursive", "section", "parent_child"}:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        # 1. Parse pages and text from file content
        pages = extract_document_pages(filename, file_bytes)
        if not pages:
            raise ValueError(f"No extractable text found in '{filename}'.")

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        file_type = detect_file_type(filename)
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        metadata = custom_metadata or {}

        # 2. Chunk according to user-selected strategy
        chunks_created = 0
        total_tokens = 0

        if strategy == "parent_child":
            pc_result = self.parent_child_chunker.split_documents(
                normalize_pages(pages, filename, metadata), doc_id=doc_id,
            )

            # Store in parent-child retriever if available, otherwise hybrid store
            if self.parent_child_retriever:
                self.parent_child_retriever.add_parent_child_result(pc_result)
            elif self.hybrid_retriever:
                # Fallback: index child chunks directly into hybrid store
                self.hybrid_retriever.add_chunks(pc_result.children)

            chunks_created = len(pc_result.children)
            total_tokens = sum(c.metadata.token_count for c in pc_result.children)

        elif strategy == "section":
            chunks = self.section_chunker.split_documents(
                documents=normalize_pages(pages, filename, metadata),
                doc_id=doc_id,
            )
            if self.hybrid_retriever:
                self.hybrid_retriever.add_chunks(chunks)

            chunks_created = len(chunks)
            total_tokens = sum(c.metadata.token_count for c in chunks)

        else:
            # Default: Recursive Character Splitting (500 tokens, 50 overlap)
            chunks = self.recursive_chunker.split_documents(
                documents=normalize_pages(pages, filename, metadata),
                doc_id=doc_id,
            )
            if self.hybrid_retriever:
                self.hybrid_retriever.add_chunks(chunks)

            chunks_created = len(chunks)
            total_tokens = sum(c.metadata.token_count for c in chunks)

        return IngestionSummary(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            total_pages=len(pages),
            strategy=strategy,
            chunk_count=chunks_created,
            total_tokens=total_tokens,
            created_at=created_at,
        )

