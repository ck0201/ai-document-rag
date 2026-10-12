"""
End-to-End RAG Pipeline Orchestrator.

Glues together:
- Document Parser & Chunker
- Embedding Engine
- Vector Store
- Prompt Builder
- LLM Client
"""

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict, Generator, List, Optional

from src.parser import DocumentParser
from src.chunker import RecursiveCharacterChunker
from src.embeddings import BaseEmbedder, FastEmbedLocal
from src.vector_store import BaseVectorStore, ChromaVectorStore, SearchResult
from src.pgvector_store import PGVectorStore
from src.prompt_builder import SYSTEM_PROMPT, build_rag_prompt
from src.llm import BaseLLMClient, get_llm_client


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    context_chunks_used: int


class RAGPipeline:
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        llm: Optional[BaseLLMClient] = None,
        chunk_size: int = 700,
        chunk_overlap: int = 150,
    ):
        self.chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.parser = DocumentParser(self.chunker)
        self.embedder = embedder or FastEmbedLocal()

        if vector_store:
            self.vector_store = vector_store
        else:
            store_type = os.environ.get("VECTOR_STORE_TYPE", "pgvector").lower()
            if store_type == "pgvector":
                try:
                    self.vector_store = PGVectorStore()
                except Exception:
                    # Fallback to Chroma if Postgres container not connected
                    self.vector_store = ChromaVectorStore()
            else:
                self.vector_store = ChromaVectorStore()

        self.llm = llm or get_llm_client()

    def ingest_document(self, file_path: str | Path, source_name: Optional[str] = None) -> int:
        """Parses, chunks, embeds, and indexes a document."""
        chunks = self.parser.parse_and_chunk(file_path, source_name=source_name)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.35,
    ) -> RAGResponse:
        """Runs the query retrieval and generation pipeline."""
        # 1. Embed Query
        q_vec = self.embedder.embed_query(question)

        # 2. Retrieve Top-K Chunks
        search_results = self.vector_store.search(q_vec, top_k=top_k)

        # 3. Filter by similarity threshold
        filtered_results = [r for r in search_results if r.score >= min_similarity]

        # 4. If nothing relevant is found, short-circuit gracefully
        if not filtered_results:
            return RAGResponse(
                query=question,
                answer="I cannot answer this question based on the provided documents.",
                sources=[],
                context_chunks_used=0,
            )

        # 5. Build Grounded Prompt
        prompt = build_rag_prompt(question, filtered_results)

        # 6. Generate LLM Answer
        answer = self.llm.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)

        # 7. Collect Source Citations
        sources = [
            {
                "source": r.chunk.metadata.get("source"),
                "page": r.chunk.metadata.get("page", 1),
                "chunk_index": r.chunk.metadata.get("chunk_index"),
                "similarity": round(r.score, 4),
            }
            for r in filtered_results
        ]

        return RAGResponse(
            query=question,
            answer=answer.strip(),
            sources=sources,
            context_chunks_used=len(filtered_results),
        )

    def query_stream(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.35,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams retrieval metadata followed by real-time token events."""
        q_vec = self.embedder.embed_query(question)
        search_results = self.vector_store.search(q_vec, top_k=top_k)
        filtered_results = [r for r in search_results if r.score >= min_similarity]

        if not filtered_results:
            yield {
                "event": "error",
                "data": {"message": "I cannot answer this question based on the provided documents."},
            }
            yield {"event": "done", "data": {}}
            return

        sources = [
            {
                "source": r.chunk.metadata.get("source"),
                "page": r.chunk.metadata.get("page", 1),
                "chunk_index": r.chunk.metadata.get("chunk_index"),
                "similarity": round(r.score, 4),
            }
            for r in filtered_results
        ]

        # 1. Send sources metadata event first
        yield {"event": "sources", "data": sources}

        prompt = build_rag_prompt(question, filtered_results)
        full_answer = self.llm.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT).strip()

        # 2. Stream tokens (words with punctuation)
        import time
        words = full_answer.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield {"event": "token", "data": {"token": token}}
            time.sleep(0.015)  # 15ms pacing to emulate real-time generation

        # 3. Final completion event
        yield {"event": "done", "data": {"status": "complete"}}
