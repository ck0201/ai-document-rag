"""
Document Parser supporting Markdown, Plaintext, and PDF files.
Extracts text with page/section level metadata.
"""

import sys
from pathlib import Path
from typing import List, Optional
import pymupdf

# Support execution from both repo root and src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chunker import Chunk, RecursiveCharacterChunker


class DocumentParser:
    def __init__(self, chunker: RecursiveCharacterChunker):
        self.chunker = chunker

    def parse_and_chunk(
        self,
        file_path: str | Path,
        source_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Chunk]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logical_name = source_name or path.name
        suffix = path.suffix.lower()
        if suffix in [".txt", ".md"]:
            return self._parse_text_file(path, logical_name, user_id=user_id)
        elif suffix == ".pdf":
            return self._parse_pdf_file(path, logical_name, user_id=user_id)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Supported: .txt, .md, .pdf")

    def _parse_text_file(self, path: Path, logical_name: str, user_id: Optional[str] = None) -> List[Chunk]:
        content = path.read_text(encoding="utf-8")
        chunks = self.chunker.chunk_document(content, source_name=logical_name, user_id=user_id)
        # For single text/md files, default page is 1
        for c in chunks:
            c.metadata["source"] = logical_name
            c.metadata["page"] = 1
            if user_id:
                c.metadata["user_id"] = user_id
        return chunks

    def _parse_pdf_file(self, path: Path, logical_name: str, user_id: Optional[str] = None) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        doc = pymupdf.open(path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if not text.strip():
                    continue

                page_chunks = self.chunker.chunk_document(
                    text,
                    source_name=f"{logical_name}#page={page_num + 1}",
                    user_id=user_id,
                )
                for c in page_chunks:
                    c.metadata["source"] = logical_name
                    c.metadata["page"] = page_num + 1
                    if user_id:
                        c.metadata["user_id"] = user_id
                    all_chunks.append(c)
        finally:
            doc.close()

        return all_chunks


if __name__ == "__main__":
    chunker = RecursiveCharacterChunker(chunk_size=700, chunk_overlap=150)
    parser = DocumentParser(chunker)

    test_file = Path("data/company_handbook.md")
    chunks = parser.parse_and_chunk(test_file)
    print(f"Parsed {len(chunks)} chunks from {test_file.name}")
    print(f"Sample Chunk 0 Metadata: {chunks[0].metadata}")
