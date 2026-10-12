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
from src.llm import BaseLLMClient, MockLLMClient, get_llm_client
from src.conversational import ConversationalRouter
import json


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
        self._suggestions_cache: Dict[str, List[Dict[str, str]]] = {}

    def clear_suggestions_cache(self) -> None:
        """Clears cached dynamic suggestions when documents change."""
        self._suggestions_cache.clear()

    def ingest_document(self, file_path: str | Path, source_name: Optional[str] = None) -> int:
        """Parses, chunks, embeds, and indexes a document."""
        chunks = self.parser.parse_and_chunk(file_path, source_name=source_name)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        self.clear_suggestions_cache()
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.35,
    ) -> RAGResponse:
        """Runs the query retrieval and generation pipeline."""
        # Check for conversational pleasantries (greetings, gratitude, small-talk)
        is_chitchat, chitchat_resp, search_question = ConversationalRouter.classify_and_respond(question)
        if is_chitchat:
            return RAGResponse(
                query=question,
                answer=chitchat_resp,
                sources=[],
                context_chunks_used=0,
            )

        # 1. Embed Query
        q_vec = self.embedder.embed_query(search_question)

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
        prompt = build_rag_prompt(search_question, filtered_results)

        # 6. Generate LLM Answer
        answer = self.llm.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)

        # 7. Collect Source Citations
        sources = [
            {
                "source": r.chunk.metadata.get("source"),
                "page": r.chunk.metadata.get("page", 1),
                "chunk_index": r.chunk.metadata.get("chunk_index"),
                "similarity": round(r.score, 4),
                "text": r.chunk.text.strip(),
                "char_start": r.chunk.metadata.get("char_start", 0),
                "char_end": r.chunk.metadata.get("char_end", 0),
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
        # Check for conversational pleasantries (greetings, gratitude, small-talk)
        is_chitchat, chitchat_resp, search_question = ConversationalRouter.classify_and_respond(question)
        if is_chitchat:
            yield {"event": "sources", "data": []}
            words = chitchat_resp.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield {"event": "token", "data": {"token": token}}
            yield {"event": "done", "data": {"status": "complete"}}
            return

        q_vec = self.embedder.embed_query(search_question)
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
                "text": r.chunk.text.strip(),
                "char_start": r.chunk.metadata.get("char_start", 0),
                "char_end": r.chunk.metadata.get("char_end", 0),
            }
            for r in filtered_results
        ]

        # 1. Send sources metadata event first
        yield {"event": "sources", "data": sources}

        prompt = build_rag_prompt(search_question, filtered_results)

        # 2. Stream tokens directly from the LLM client
        for token in self.llm.generate_stream(prompt=prompt, system_instruction=SYSTEM_PROMPT):
            yield {"event": "token", "data": {"token": token}}

        # 3. Final completion event
        yield {"event": "done", "data": {"status": "complete"}}

    def get_suggestions(self) -> List[Dict[str, str]]:
        """
        Dynamically generates or retrieves smart prompt suggestions
        tailored to the currently indexed documents.
        """
        docs = []
        if hasattr(self.vector_store, "list_documents"):
            docs = self.vector_store.list_documents()

        if not docs:
            return [
                {
                    "icon": "📄",
                    "title": "Upload a Document",
                    "subtitle": "Drag & drop PDF, Markdown, or text files",
                    "query": "What types of documents and formats are supported?",
                },
                {
                    "icon": "🔒",
                    "title": "Security & Privacy",
                    "subtitle": "Automatic PII masking and prompt injection defense",
                    "query": "How does the assistant protect confidential data and redact PII?",
                },
                {
                    "icon": "⚡",
                    "title": "Hybrid Retrieval",
                    "subtitle": "PostgreSQL HNSW vector search + BM25 sparse matching",
                    "query": "How does hybrid retrieval and reranking work in this system?",
                },
                {
                    "icon": "💡",
                    "title": "Assistant Capabilities",
                    "subtitle": "Discover enterprise Q&A intelligence features",
                    "query": "What can you do and how do you cite sources?",
                },
            ]

        # Cache key based on document names and chunk counts
        cache_key = "|".join(f"{d.get('source')}:{d.get('chunks_count')}" for d in docs)
        if cache_key in self._suggestions_cache:
            return self._suggestions_cache[cache_key]

        # Sample snippet text from the indexed documents
        sample_snippets = []
        for d in docs[:3]:
            src_name = d.get("source", "")
            if hasattr(self.vector_store, "_get_connection"):
                try:
                    with self.vector_store._get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT text FROM document_chunks WHERE source = %s ORDER BY id ASC LIMIT 2;",
                                (src_name,),
                            )
                            for row in cur.fetchall():
                                sample_snippets.append(f"[{src_name}]: {row[0][:450]}")
                except Exception:
                    pass
            elif hasattr(self.vector_store, "collection"):
                try:
                    chroma_res = self.vector_store.collection.get(
                        where={"source": src_name},
                        limit=2,
                        include=["documents"],
                    )
                    for doc_text in chroma_res.get("documents", []):
                        sample_snippets.append(f"[{src_name}]: {doc_text[:450]}")
                except Exception:
                    pass

        # If we have snippets and a real LLM, ask the LLM to generate 4 high-value questions
        if sample_snippets and self.llm and not isinstance(self.llm, MockLLMClient):
            context_text = "\n---\n".join(sample_snippets)
            prompt = (
                "You are an AI generating 4 recommended starter questions for a user exploring these documents.\n"
                f"EXCERPTS:\n{context_text}\n\n"
                "Instructions:\n"
                "- Generate exactly 4 diverse, insightful questions that can be answered from this content.\n"
                "- Return ONLY valid JSON array with 4 objects. No markdown ticks, no commentary.\n"
                "- Each object must have:\n"
                '  "icon": a relevant single emoji\n'
                '  "title": category label (2-3 words)\n'
                '  "subtitle": quick summary of the card (4-8 words)\n'
                '  "query": the natural question to ask\n'
            )
            try:
                raw_json = self.llm.generate(
                    prompt=prompt,
                    system_instruction="You are a JSON generator. Output only a valid JSON array.",
                ).strip()
                if raw_json.startswith("```"):
                    raw_json = raw_json.split("\n", 1)[1]
                    if raw_json.endswith("```"):
                        raw_json = raw_json.rsplit("```", 1)[0]
                parsed = json.loads(raw_json.strip())
                if isinstance(parsed, list) and len(parsed) >= 2:
                    formatted = []
                    for item in parsed[:4]:
                        formatted.append({
                            "icon": str(item.get("icon", "📄")),
                            "title": str(item.get("title", "Document Overview")),
                            "subtitle": str(item.get("subtitle", "Explore key details")),
                            "query": str(item.get("query", "")),
                        })
                    self._suggestions_cache[cache_key] = formatted
                    return formatted
            except Exception:
                pass

        # Fallback heuristic suggestions based on document names
        primary_doc = docs[0].get("source", "Document")
        fallback_suggestions = [
            {
                "icon": "📋",
                "title": "Document Summary",
                "subtitle": f"Key takeaways from {primary_doc}",
                "query": f"What is the main topic and summary of {primary_doc}?",
            },
            {
                "icon": "🔍",
                "title": "Key Responsibilities",
                "subtitle": "Roles, milestones and achievements",
                "query": f"What are the key roles, responsibilities, or milestones mentioned in {primary_doc}?",
            },
            {
                "icon": "⚙️",
                "title": "Technical Details",
                "subtitle": "Technologies, tools, or specifications",
                "query": f"What technologies, tools, or specifications are listed in {primary_doc}?",
            },
            {
                "icon": "🎯",
                "title": "Specific Highlights",
                "subtitle": "Notable achievements or requirements",
                "query": f"What notable achievements, requirements, or highlights are described in {primary_doc}?",
            },
        ]
        self._suggestions_cache[cache_key] = fallback_suggestions
        return fallback_suggestions
