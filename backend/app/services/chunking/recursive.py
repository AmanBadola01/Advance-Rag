"""
Recursive Character Splitting Chunker
Uses LangChain's RecursiveCharacterTextSplitter with token-aware splitting.
"""

import uuid
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import ProcessedChunk
from .base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Recursive Character Splitting:
    Uses LangChain's RecursiveCharacterTextSplitter.
    Configured for 500 tokens chunk size and 50 tokens overlap.
    Splits along paragraph (\\n\\n) -> sentence (\\n) -> word (space) boundaries for clean breaks.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
        separators: Optional[List[str]] = None,
    ):
        super().__init__(encoding_name=encoding_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        # Initialize LangChain RecursiveCharacterTextSplitter with token-based length function
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens,
            separators=self.separators,
            keep_separator=True,
        )

    def split_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source: str = "document.txt",
        page: Optional[int] = 1,
        file_type: Optional[str] = None,
        section: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedChunk]:
        """Split a raw text string into chunks."""
        if not text or not text.strip():
            return []

        doc_id = doc_id or str(uuid.uuid4())[:8]
        raw_splits = self.splitter.split_text(text)

        chunks: List[ProcessedChunk] = []
        for idx, split_text in enumerate(raw_splits):
            chunk = self.create_chunk(
                text=split_text.strip(),
                index=idx,
                strategy="recursive",
                doc_id=doc_id,
                source=source,
                page=page,
                file_type=file_type,
                section=section,
                custom_metadata=custom_metadata,
            )
            chunks.append(chunk)

        return chunks

    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
    ) -> List[ProcessedChunk]:
        """
        Split a list of document pages/sections.
        Each document item should have 'text' and optional 'metadata' dict.
        """
        doc_id = doc_id or str(uuid.uuid4())[:8]
        all_chunks: List[ProcessedChunk] = []
        chunk_counter = 0

        for doc in documents:
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            source = meta.get("source", "document.txt")
            page = meta.get("page", 1)
            file_type = meta.get("file_type", None)
            section = meta.get("section", None)

            raw_splits = self.splitter.split_text(text)
            for split_text in raw_splits:
                if not split_text.strip():
                    continue
                chunk = self.create_chunk(
                    text=split_text.strip(),
                    index=chunk_counter,
                    strategy="recursive",
                    doc_id=doc_id,
                    source=source,
                    page=page,
                    file_type=file_type,
                    section=section,
                    custom_metadata=meta,
                )
                all_chunks.append(chunk)
                chunk_counter += 1

        return all_chunks

