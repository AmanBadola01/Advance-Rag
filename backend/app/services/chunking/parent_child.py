"""
Parent-Child Chunking Strategy
Creates small child chunks (~200 tokens) for vector retrieval precision,
linked directly to larger parent chunks (~1000 tokens) that provide rich context to the LLM.
"""

import uuid
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import ProcessedChunk, ParentChildResult
from .base import BaseChunker


class ParentChildChunker(BaseChunker):
    """
    Parent-Child Chunking:
    1. Parent Chunks: 1000 tokens (large context window for LLM synthesis).
    2. Child Chunks: 200 tokens (small, focused chunks for vector similarity search).
    3. Linkage: Each child chunk stores `parent_id` pointing to its parent chunk.
    4. Retrieval Resolution: Vector database searches by child chunks, but retrieves parent chunks for LLM context.
    """

    # Initializes parent and child text splitters with 1000-token and 200-token target sizes.
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 25,
        encoding_name: str = "cl100k_base",
    ):
        """
        Configure parent and child text splitters.
        Uses token-aware splitting based on tiktoken cl100k_base.
        """
        super().__init__(encoding_name=encoding_name)
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

        # Splitter for large parent context chunks (~1000 tokens)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        )

        # Splitter for small child retrieval chunks (~200 tokens)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        )

    # Splits text into 1000-token parent chunks and sub-splits each into 200-token linked child chunks.
    def split_into_parent_child(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source: str = "document.txt",
        page: Optional[int] = 1,
        file_type: Optional[str] = None,
        section: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> ParentChildResult:
        """
        Execute hierarchical two-tier splitting:
        Returns ParentChildResult containing both parent chunks and linked child chunks.
        """
        if not text or not text.strip():
            return ParentChildResult(parents=[], children=[])

        doc_id = doc_id or str(uuid.uuid4())[:8]
        raw_parents = self.parent_splitter.split_text(text)

        parent_chunks: List[ProcessedChunk] = []
        child_chunks: List[ProcessedChunk] = []
        child_counter = 0

        for p_idx, p_text in enumerate(raw_parents):
            clean_parent_text = p_text.strip()
            if not clean_parent_text:
                continue

            parent_id = f"{doc_id}_parent_{p_idx}"
            
            # Construct Parent Chunk (is_parent=True, 1000 tokens)
            parent_chunk = self.create_chunk(
                text=clean_parent_text,
                index=p_idx,
                strategy="parent_child",
                doc_id=doc_id,
                source=source,
                page=page,
                file_type=file_type,
                section=section,
                parent_id=None,
                is_parent=True,
                custom_metadata=dict(custom_metadata or {}),
            )
            # Ensure chunk_id explicitly reflects parent identity
            parent_chunk.chunk_id = parent_id
            parent_chunk.metadata.chunk_id = parent_id
            parent_chunks.append(parent_chunk)

            # Sub-split this parent into smaller child chunks (~200 tokens)
            raw_children = self.child_splitter.split_text(clean_parent_text)
            for c_text in raw_children:
                clean_child_text = c_text.strip()
                if not clean_child_text:
                    continue

                child_id = f"{doc_id}_child_{child_counter}"
                child_meta = dict(custom_metadata or {})
                child_meta["parent_chunk_index"] = p_idx

                # Construct Child Chunk (is_parent=False, linked via parent_id)
                child_chunk = self.create_chunk(
                    text=clean_child_text,
                    index=child_counter,
                    strategy="parent_child",
                    doc_id=doc_id,
                    source=source,
                    page=page,
                    file_type=file_type,
                    section=section,
                    parent_id=parent_id,
                    is_parent=False,
                    custom_metadata=child_meta,
                )
                child_chunk.chunk_id = child_id
                child_chunk.metadata.chunk_id = child_id
                child_chunks.append(child_chunk)
                child_counter += 1

        return ParentChildResult(parents=parent_chunks, children=child_chunks)

    # Standard chunker interface returning retrieval-ready child chunks stamped with their parent IDs.
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
        """
        Standard chunker interface for vector store ingestion.
        Returns the child chunks (which get embedded in ChromaDB).
        """
        result = self.split_into_parent_child(
            text=text,
            doc_id=doc_id,
            source=source,
            page=page,
            file_type=file_type,
            section=section,
            custom_metadata=custom_metadata,
        )
        return result.children

    # Batches multi-page documents into parent and child chunks across all document pages.
    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
    ) -> ParentChildResult:
        """
        Batch process multi-page documents (e.g. from PyPDFLoader).
        Preserves sequential parent and child chunk indices across all pages.
        """
        doc_id = doc_id or str(uuid.uuid4())[:8]
        all_parents: List[ProcessedChunk] = []
        all_children: List[ProcessedChunk] = []

        parent_counter = 0
        child_counter = 0

        for doc in documents:
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            source = meta.get("source", "document.txt")
            page = meta.get("page", 1)
            file_type = meta.get("file_type", None)
            section = meta.get("section", None)

            page_result = self.split_into_parent_child(
                text=text,
                doc_id=doc_id,
                source=source,
                page=page,
                file_type=file_type,
                section=section,
                custom_metadata=meta,
            )

            # Re-index to ensure globally sequential IDs across document pages
            parent_id_mapping = {}
            for p in page_result.parents:
                old_id = p.chunk_id
                new_id = f"{doc_id}_parent_{parent_counter}"
                parent_id_mapping[old_id] = new_id
                p.chunk_id = new_id
                p.metadata.chunk_id = new_id
                p.metadata.chunk_index = parent_counter
                all_parents.append(p)
                parent_counter += 1

            for c in page_result.children:
                old_parent_id = c.metadata.parent_id
                c.metadata.parent_id = parent_id_mapping.get(old_parent_id, old_parent_id)
                new_child_id = f"{doc_id}_child_{child_counter}"
                c.chunk_id = new_child_id
                c.metadata.chunk_id = new_child_id
                c.metadata.chunk_index = child_counter
                all_children.append(c)
                child_counter += 1

        return ParentChildResult(parents=all_parents, children=all_children)

    # Resolves and returns the 1000-token parent chunk for a given retrieved child chunk.
    def resolve_parent(
        self,
        child_chunk: ProcessedChunk,
        parent_store: Dict[str, ProcessedChunk],
    ) -> ProcessedChunk:
        """
        Core retrieval helper:
        Given a child chunk retrieved from vector similarity search,
        looks up and returns its corresponding parent chunk to send to the LLM.
        Falls back to child chunk if parent is missing.
        """
        parent_id = child_chunk.metadata.parent_id
        if parent_id and parent_id in parent_store:
            return parent_store[parent_id]
        return child_chunk

