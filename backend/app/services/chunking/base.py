"""
Base Chunker
Provides shared token counting and standardized ProcessedChunk construction.
"""

from typing import Dict, Any, Optional
import tiktoken
from app.models.schemas import ProcessedChunk, ChunkMetadata


class BaseChunker:
    """Base class for all chunking strategies."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken with fallback."""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback approximation: ~4 characters per token
        return max(1, len(text) // 4)

    def create_chunk(
        self,
        text: str,
        index: int,
        strategy: str,
        doc_id: str,
        source: str,
        page: Optional[int] = 1,
        file_type: Optional[str] = None,
        section: Optional[str] = None,
        parent_id: Optional[str] = None,
        is_parent: bool = False,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessedChunk:
        """Construct a standardized ProcessedChunk instance."""
        chunk_id = f"{doc_id}_{strategy}_{index}"
        token_count = self.count_tokens(text)
        char_count = len(text)
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            chunk_index=index,
            strategy=strategy,
            source=source,
            page=page,
            file_type=file_type,
            token_count=token_count,
            char_count=char_count,
            section=section,
            parent_id=parent_id,
            is_parent=is_parent,
            custom_metadata=custom_metadata or {},
        )
        return ProcessedChunk(chunk_id=chunk_id, text=text, metadata=metadata)

