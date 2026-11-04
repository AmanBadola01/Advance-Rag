"""
Section-Based Chunker
Splits documents along detected H1, H2, H3 headers using LangChain's MarkdownHeaderTextSplitter.
"""

import re
import uuid
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from app.models.schemas import ProcessedChunk
from .base import BaseChunker


class SectionBasedChunker(BaseChunker):
    """
    Section-Based Chunking:
    Splits documents along detected section headers (H1, H2, H3).
    Extracts header text as section metadata to enable section filtering (e.g., 'Financial Summary').
    If a section exceeds max_section_tokens, sub-splits it recursively while retaining section metadata.
    """

    # Initializes the section chunker with target headers, token thresholds, splitters, and regex patterns.
    def __init__(
        self,
        headers_to_split_on: Optional[List[tuple]] = None,
        max_section_tokens: int = 800,
        sub_split_large_sections: bool = True,
        encoding_name: str = "cl100k_base",
    ):
        super().__init__(encoding_name=encoding_name)
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.max_section_tokens = max_section_tokens
        self.sub_split_large_sections = sub_split_large_sections
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_section_tokens,
            chunk_overlap=50,
            length_function=self.count_tokens,
        )
        self.header_regex = re.compile(
            r"^((\d+(\.\d+)*\s+[A-Z][A-Za-z0-9\s,\-]{2,60})|"
            r"(SECTION\s+[\dA-Z]+[:\-\s]+[A-Za-z0-9\s,\-]{2,60})|"
            r"(CHAPTER\s+[\dA-Z]+[:\-\s]+[A-Za-z0-9\s,\-]{2,60})|"
            r"(APPENDIX\s+[\dA-Z]+[:\-\s]+[A-Za-z0-9\s,\-]{2,60}))$",
            re.IGNORECASE,
        )

    # Detects plain-text numbered/capitalized headings and converts them to markdown headers (##).
    def _normalize_plain_headers(self, text: str) -> str:
        """
        Detect plain-text section headers (e.g. '1.0 Introduction', 'SECTION 2:')
        and prepend '## ' if Markdown headers are not already present.
        """
        if re.search(r"^#{1,3}\s+", text, re.MULTILINE):
            return text

        lines = text.split("\n")
        normalized: List[str] = []
        for line in lines:
            stripped = line.strip()
            if self.header_regex.match(stripped):
                normalized.append(f"## {stripped}")
            else:
                normalized.append(line)
        return "\n".join(normalized)

    # Extracts the most specific section title (Header 3 -> Header 2 -> Header 1) from LangChain metadata.
    def _extract_primary_section(self, header_meta: Dict[str, str]) -> str:
        """Extract the most specific section title available in metadata."""
        for key in ["Header 3", "Header 2", "Header 1"]:
            if key in header_meta and header_meta[key]:
                return header_meta[key].strip()
        return "General"

    # Splits raw text into section-aware chunks and sub-splits sections exceeding the token limit.
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
        """Split document by detected section headers."""
        if not text or not text.strip():
            return []

        doc_id = doc_id or str(uuid.uuid4())[:8]
        normalized_text = self._normalize_plain_headers(text)
        md_docs = self.md_splitter.split_text(normalized_text)

        # Fallback if no markdown headers were detected
        if not md_docs:
            md_docs = [type("Doc", (), {"page_content": text, "metadata": {}})]

        chunks: List[ProcessedChunk] = []
        chunk_idx = 0

        for doc in md_docs:
            content = doc.page_content.strip()
            if not content:
                continue

            extracted_section = section or self._extract_primary_section(doc.metadata)
            meta_dict = dict(custom_metadata or {})
            if doc.metadata:
                meta_dict["header_hierarchy"] = doc.metadata

            tokens = self.count_tokens(content)
            if self.sub_split_large_sections and tokens > self.max_section_tokens:
                sub_splits = self.sub_splitter.split_text(content)
                for part_idx, sub_text in enumerate(sub_splits):
                    sub_meta = dict(meta_dict)
                    sub_meta["part_index"] = part_idx + 1
                    sub_meta["total_parts"] = len(sub_splits)
                    chunk = self.create_chunk(
                        text=sub_text.strip(),
                        index=chunk_idx,
                        strategy="section",
                        doc_id=doc_id,
                        source=source,
                        page=page,
                        file_type=file_type,
                        section=extracted_section,
                        custom_metadata=sub_meta,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
            else:
                chunk = self.create_chunk(
                    text=content,
                    index=chunk_idx,
                    strategy="section",
                    doc_id=doc_id,
                    source=source,
                    page=page,
                    file_type=file_type,
                    section=extracted_section,
                    custom_metadata=meta_dict,
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks

    # Processes multi-page document batches while preserving section hierarchy and sequential chunk IDs.
    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
    ) -> List[ProcessedChunk]:
        """Split a list of documents or pages preserving section headers."""
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

            doc_chunks = self.split_text(
                text=text,
                doc_id=doc_id,
                source=source,
                page=page,
                file_type=file_type,
                section=section,
                custom_metadata=meta,
            )
            for c in doc_chunks:
                # Re-index chunks to ensure strict sequential ordering
                c.metadata.chunk_index = chunk_counter
                c.metadata.chunk_id = f"{doc_id}_section_{chunk_counter}"
                all_chunks.append(c)
                chunk_counter += 1

        return all_chunks

