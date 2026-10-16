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
from src.hybrid_retriever import HybridRetriever
from src.prompt_builder import SYSTEM_PROMPT, build_rag_prompt
from src.llm import BaseLLMClient, MockLLMClient, get_llm_client
from src.conversational import ConversationalRouter
import json
import time

from src.memory import ConversationSession, QueryReformulator, SessionManager, default_session_manager


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    context_chunks_used: int
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    groundedness_score: float = 1.0
    vector_store_type: str = "pgvector (PostgreSQL HNSW)"


class RAGPipeline:
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        llm: Optional[BaseLLMClient] = None,
        session_manager: Optional[SessionManager] = None,
        chunk_size: int = 900,
        chunk_overlap: int = 200,
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

        self.hybrid_retriever = HybridRetriever(self.vector_store, self.embedder)
        self.llm = llm or get_llm_client()
        self.session_manager = session_manager or default_session_manager
        self.reformulator = QueryReformulator()
        self._suggestions_cache: Dict[str, List[Dict[str, str]]] = {}

    def _get_store_label(self) -> str:
        """Returns human-readable vector store engine label."""
        if hasattr(self.vector_store, "__class__") and "PGVector" in self.vector_store.__class__.__name__:
            return "pgvector (PostgreSQL HNSW)"
        return "ChromaDB (HNSW)"

    def clear_suggestions_cache(self) -> None:
        """Clears cached dynamic suggestions when documents change."""
        self._suggestions_cache.clear()

    def _build_filter_metadata(self, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Constructs vector search filter for multi-tenant isolation."""
        if not user_id:
            return None

        filter_meta: Dict[str, Any] = {"user_id": user_id}
        user_docs_count = 0
        if hasattr(self.vector_store, "count_user_documents"):
            user_docs_count = self.vector_store.count_user_documents(user_id)
        elif hasattr(self.vector_store, "list_documents"):
            docs = self.vector_store.list_documents(user_id=user_id)
            user_docs_count = len([d for d in docs if not d.get("is_sample")])

        filter_meta["allow_global_fallback"] = (user_docs_count == 0)
        return filter_meta

    def ingest_document(
        self,
        file_path: str | Path,
        source_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Parses, chunks, embeds, and indexes a document with optional user_id isolation."""
        chunks = self.parser.parse_and_chunk(file_path, source_name=source_name, user_id=user_id)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        self.hybrid_retriever.invalidate_bm25(user_id=user_id)
        self.clear_suggestions_cache()
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: float = 0.20,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> RAGResponse:
        """Runs the query retrieval and generation pipeline with conversational memory & latency telemetry."""
        t_start = time.perf_counter()
        session = self.session_manager.get_or_create(session_id) if session_id else None
        store_label = self._get_store_label()

        # Check for conversational pleasantries (greetings, gratitude, small-talk)
        is_chitchat, chitchat_resp, search_question = ConversationalRouter.classify_and_respond(question)
        if is_chitchat:
            if session:
                session.add_user_message(question)
                session.add_assistant_message(chitchat_resp)
            total_ms = (time.perf_counter() - t_start) * 1000
            return RAGResponse(
                query=question,
                answer=chitchat_resp,
                sources=[],
                context_chunks_used=0,
                retrieval_latency_ms=0.0,
                generation_latency_ms=round(total_ms, 2),
                total_latency_ms=round(total_ms, 2),
                groundedness_score=1.0,
                vector_store_type=store_label,
            )

        # 1. Multi-turn context resolution: Reformulate ambiguous queries if context exists
        standalone_search_query = self.reformulator.reformulate(search_question, session)

        # 2. Retrieve candidates via Hybrid Search (Dense + BM25 + FlashRank Reranker)
        t_retrieval_start = time.perf_counter()
        filter_metadata = self._build_filter_metadata(user_id)
        search_results = self.hybrid_retriever.retrieve(
            query=standalone_search_query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            user_id=user_id,
        )
        filtered_results = [r for r in search_results if r.score >= min_similarity]
        if not filtered_results and search_results:
            filtered_results = [r for r in search_results if r.score >= 0.05]
        retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

        # 3. If nothing relevant is found, short-circuit gracefully
        if not filtered_results:
            refusal = "I cannot answer this question based on the provided documents."
            if session:
                session.add_user_message(question)
                session.add_assistant_message(refusal)
            total_ms = (time.perf_counter() - t_start) * 1000
            return RAGResponse(
                query=question,
                answer=refusal,
                sources=[],
                context_chunks_used=0,
                retrieval_latency_ms=round(retrieval_ms, 2),
                generation_latency_ms=0.0,
                total_latency_ms=round(total_ms, 2),
                groundedness_score=0.0,
                vector_store_type=store_label,
            )

        # 4. Build Grounded Prompt with Conversation History
        conv_summary = session.get_context_summary(max_turns=3) if session else ""
        prompt = build_rag_prompt(
            question=search_question,
            search_results=filtered_results,
            conversation_history=conv_summary,
        )

        # 5. Generate LLM Answer
        t_gen_start = time.perf_counter()
        answer = self.llm.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT).strip()
        generation_ms = (time.perf_counter() - t_gen_start) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        # 6. Record turns in session
        if session:
            session.add_user_message(question)
            session.add_assistant_message(answer)

        # 7. Collect Source Citations & Calculate Confidence
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

        top_similarity = max((s["similarity"] for s in sources), default=0.0)
        groundedness = 1.0 if top_similarity >= 0.55 else round(min(1.0, max(0.4, top_similarity / 0.65)), 2)

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            context_chunks_used=len(filtered_results),
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
            total_latency_ms=round(total_ms, 2),
            groundedness_score=groundedness,
            vector_store_type=store_label,
        )

    def query_stream(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: float = 0.20,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams retrieval metadata followed by real-time token events and observability breakdown."""
        t_start = time.perf_counter()
        session = self.session_manager.get_or_create(session_id) if session_id else None
        store_label = self._get_store_label()

        # Check for conversational pleasantries (greetings, gratitude, small-talk)
        is_chitchat, chitchat_resp, search_question = ConversationalRouter.classify_and_respond(question)
        if is_chitchat:
            yield {"event": "sources", "data": [], "retrieval_ms": 0.0}
            words = chitchat_resp.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield {"event": "token", "data": {"token": token}}
            if session:
                session.add_user_message(question)
                session.add_assistant_message(chitchat_resp)
            total_ms = (time.perf_counter() - t_start) * 1000
            yield {
                "event": "done",
                "data": {
                    "status": "complete",
                    "latency": {
                        "retrieval_ms": 0.0,
                        "generation_ms": round(total_ms, 1),
                        "total_ms": round(total_ms, 1),
                    },
                    "vector_store": store_label,
                    "groundedness": 1.0,
                },
            }
            return

        # 1. Multi-turn query reformulation
        standalone_search_query = self.reformulator.reformulate(search_question, session)

        # 2. Retrieve candidates via Hybrid Search (Dense + BM25 + FlashRank Reranker)
        t_retrieval_start = time.perf_counter()
        filter_metadata = self._build_filter_metadata(user_id)
        search_results = self.hybrid_retriever.retrieve(
            query=standalone_search_query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            user_id=user_id,
        )
        filtered_results = [r for r in search_results if r.score >= min_similarity]
        if not filtered_results and search_results:
            filtered_results = [r for r in search_results if r.score >= 0.05]
        retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

        if not filtered_results:
            refusal = "I cannot answer this question based on the provided documents."
            if session:
                session.add_user_message(question)
                session.add_assistant_message(refusal)
            yield {
                "event": "error",
                "data": {"message": refusal},
            }
            total_ms = (time.perf_counter() - t_start) * 1000
            yield {
                "event": "done",
                "data": {
                    "status": "complete",
                    "latency": {
                        "retrieval_ms": round(retrieval_ms, 1),
                        "generation_ms": 0.0,
                        "total_ms": round(total_ms, 1),
                    },
                    "vector_store": store_label,
                    "groundedness": 0.0,
                },
            }
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

        top_similarity = max((s["similarity"] for s in sources), default=0.0)
        groundedness = 1.0 if top_similarity >= 0.55 else round(min(1.0, max(0.4, top_similarity / 0.65)), 2)

        # Send sources metadata event first (including retrieval latency)
        yield {
            "event": "sources",
            "data": sources,
            "retrieval_ms": round(retrieval_ms, 1),
        }

        # Build prompt with conversation history
        conv_summary = session.get_context_summary(max_turns=3) if session else ""
        prompt = build_rag_prompt(
            question=search_question,
            search_results=filtered_results,
            conversation_history=conv_summary,
        )

        # Stream tokens and track generation latency
        t_gen_start = time.perf_counter()
        full_streamed_answer = []
        for token in self.llm.generate_stream(prompt=prompt, system_instruction=SYSTEM_PROMPT):
            full_streamed_answer.append(token)
            yield {"event": "token", "data": {"token": token}}

        generation_ms = (time.perf_counter() - t_gen_start) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        # Save turns to session
        if session:
            final_text = "".join(full_streamed_answer).strip()
            session.add_user_message(question)
            session.add_assistant_message(final_text)

        # Final completion event with detailed latency decomposition & confidence
        yield {
            "event": "done",
            "data": {
                "status": "complete",
                "latency": {
                    "retrieval_ms": round(retrieval_ms, 1),
                    "generation_ms": round(generation_ms, 1),
                    "total_ms": round(total_ms, 1),
                },
                "vector_store": store_label,
                "groundedness": groundedness,
            },
        }

    def get_suggestions(self, user_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Dynamically generates or retrieves smart prompt suggestions
        tailored to the currently indexed documents for this user workspace.
        """
        docs = []
        if hasattr(self.vector_store, "list_documents"):
            docs = self.vector_store.list_documents(user_id=user_id)

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

        # Cache key based on user_id, document names, and chunk counts
        cache_key = f"{user_id or 'all'}:" + "|".join(f"{d.get('source')}:{d.get('chunks_count')}" for d in docs)
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
