"""
Core Document Parsing & Chunking Module.

Implements recursive semantic chunking with sliding overlap and metadata enrichment.
Zero external heavy dependencies to understand the raw mechanics under the hood.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, List, Optional


@dataclass
class Chunk:
    """Represents an ingested documsent chunk ready for embedding and indexing."""
    chunk_id: str
    text: str
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ") + ("..." if len(self.text) > 60 else "")
        return (
            f"Chunk(id={self.chunk_id[:8]}..., "
            f"len={self.char_count}, "
            f"index={self.metadata.get('chunk_index')}, "
            f"preview='{preview}')"
        )


class RecursiveCharacterChunker:
    """
    Splits text hierarchically using a list of delimiters:
    1. Paragraphs ("\\n\\n")
    2. Lines / list items ("\\n")
    3. Sentences (". ")
    4. Words (" ")
    5. Raw characters ("") as final fallback.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using the highest priority separator found."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return list(text)

        if separator not in text:
            return self._split_text(text, remaining_separators)

        # Split preserving context
        parts = text.split(separator)
        good_splits: List[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                good_splits.append(part)
            else:
                # If a piece is still larger than chunk_size, recursively split using next separator
                sub_splits = self._split_text(part, remaining_separators) if remaining_separators else [part]
                good_splits.extend(sub_splits)

        return good_splits

    def chunk_document(self, text: str, source_name: str, user_id: Optional[str] = None) -> List[Chunk]:
        """
        Chunks text with sliding overlap and attaches metadata.
        Supports multi-tenant user_id isolation.
        """
        raw_pieces = self._split_text(text, self.separators)

        chunks: List[Chunk] = []
        current_pieces: List[str] = []
        current_len = 0
        chunk_idx = 0
        running_char_offset = 0

        for piece in raw_pieces:
            piece_len = len(piece)

            # Check if adding this piece exceeds chunk_size
            if current_pieces and (current_len + piece_len > self.chunk_size):
                combined_text = "\n\n".join(current_pieces).strip()

                # Emit chunk if it has substantial content
                chunk_hash = hashlib.sha256(
                    f"{user_id or 'global'}:{source_name}:{chunk_idx}:{combined_text}".encode()
                ).hexdigest()

                chunk_meta = {
                    "source": source_name,
                    "chunk_index": chunk_idx,
                    "char_start": running_char_offset,
                    "char_end": running_char_offset + len(combined_text),
                }
                if user_id:
                    chunk_meta["user_id"] = user_id

                chunks.append(
                    Chunk(
                        chunk_id=chunk_hash,
                        text=combined_text,
                        char_count=len(combined_text),
                        metadata=chunk_meta,
                    )
                )
                chunk_idx += 1
                running_char_offset += max(0, len(combined_text) - self.chunk_overlap)

                # Keep overlap pieces from the tail of current_pieces
                overlap_pieces: List[str] = []
                accumulated_overlap = 0
                for p in reversed(current_pieces):
                    if accumulated_overlap + len(p) <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        accumulated_overlap += len(p)
                    else:
                        break

                current_pieces = overlap_pieces
                current_len = sum(len(p) for p in current_pieces)

            current_pieces.append(piece)
            current_len += piece_len

        # Flush final chunk
        if current_pieces:
            combined_text = "\n\n".join(current_pieces).strip()
            if combined_text:
                chunk_hash = hashlib.sha256(
                    f"{user_id or 'global'}:{source_name}:{chunk_idx}:{combined_text}".encode()
                ).hexdigest()
                chunk_meta = {
                    "source": source_name,
                    "chunk_index": chunk_idx,
                    "char_start": running_char_offset,
                    "char_end": running_char_offset + len(combined_text),
                }
                if user_id:
                    chunk_meta["user_id"] = user_id

                chunks.append(
                    Chunk(
                        chunk_id=chunk_hash,
                        text=combined_text,
                        char_count=len(combined_text),
                        metadata=chunk_meta,
                    )
                )

        return chunks


if __name__ == "__main__":
    from pathlib import Path

    doc_path = Path("data/company_handbook.md")
    content = doc_path.read_text(encoding="utf-8")

    chunker = RecursiveCharacterChunker(chunk_size=700, chunk_overlap=150)
    result_chunks = chunker.chunk_document(content, source_name=doc_path.name)

    print(f"\n=======================================================")
    print(f"Document Size: {len(content)} characters")
    print(f"Generated Chunks: {len(result_chunks)}")
    print(f"=======================================================\n")

    for i, c in enumerate(result_chunks):
        print(f"--- [Chunk {c.metadata['chunk_index']} | {c.char_count} chars | ID: {c.chunk_id[:8]}] ---")
        print(f"Metadata: {c.metadata}")
        print(f"Text:\n{c.text}\n")
