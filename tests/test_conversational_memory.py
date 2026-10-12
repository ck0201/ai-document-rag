"""
Unit and Integration Tests for Conversational Memory & Query Reformulation.
"""

from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import ChatTurn, ConversationSession, QueryReformulator, SessionManager
from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaVectorStore
from src.embeddings import FastEmbedLocal
from src.llm import MockLLMClient
from src.chunker import RecursiveCharacterChunker
from src.parser import DocumentParser


def test_conversation_session():
    session = ConversationSession(session_id="test-session-1", max_turns=3)
    assert len(session.history) == 0

    session.add_user_message("What is the school fee?")
    session.add_assistant_message("The tuition fee is $5,000 per term.")
    assert len(session.history) == 2
    assert session.get_last_user_query() == "What is the school fee?"

    # Add more turns beyond max_turns (3 turns = 6 messages)
    for i in range(5):
        session.add_user_message(f"Question {i}")
        session.add_assistant_message(f"Answer {i}")

    # Maximum 6 messages preserved
    assert len(session.history) == 6
    assert session.history[-1].content == "Answer 4"

    # Context summary
    summary = session.get_context_summary(max_turns=2)
    assert "User: Question 3" in summary
    assert "Assistant: Answer 4" in summary

    # Clear
    session.clear()
    assert len(session.history) == 0


def test_query_reformulator():
    reformulator = QueryReformulator()
    session = ConversationSession(session_id="test-reformulate")
    session.add_user_message("What is the fee payment deadline?")
    session.add_assistant_message("The deadline is May 10, 2024.")

    # 1. Unambiguous query: should not be changed
    q1 = "What are the school bus routes and timings?"
    assert reformulator.reformulate(q1, session) == q1

    # 2. Pronoun reference: 'it'
    q2 = "Can it be paid in installments?"
    ref2 = reformulator.reformulate(q2, session)
    assert "regarding What is the fee payment deadline" in ref2
    assert "Can it be paid in installments" in ref2

    # 3. Follow-up connector: 'And what about'
    q3 = "And what about the late fee penalty?"
    ref3 = reformulator.reformulate(q3, session)
    assert "regarding What is the fee payment deadline" in ref3

    # 4. Short follow-up question
    q4 = "Who approves it?"
    ref4 = reformulator.reformulate(q4, session)
    assert "regarding What is the fee payment deadline" in ref4

    # 5. Empty session: query should be preserved untouched
    empty_session = ConversationSession(session_id="empty")
    assert reformulator.reformulate(q2, empty_session) == q2


def test_session_manager():
    mgr = SessionManager(ttl_seconds=2)
    s1 = mgr.get_or_create("user-alice")
    s2 = mgr.get_or_create("user-bob")

    s1.add_user_message("Hello from Alice")
    s2.add_user_message("Hello from Bob")

    # Isolation check
    assert len(s1.history) == 1
    assert s1.history[0].content == "Hello from Alice"
    assert len(s2.history) == 1
    assert s2.history[0].content == "Hello from Bob"

    # Deletion
    assert mgr.delete_session("user-alice") is True
    assert mgr.get_session("user-alice") is None
    assert mgr.get_session("user-bob") is not None

    # Expiry
    time.sleep(2.1)
    expired_count = mgr.cleanup_expired()
    assert expired_count >= 1
    assert mgr.count() == 0


def test_rag_pipeline_multi_turn():
    # Setup lightweight in-memory pipeline
    embedder = FastEmbedLocal()
    vector_store = ChromaVectorStore(collection_name="test_memory_col")
    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
    parser = DocumentParser(chunker)
    chunks = parser.parse_and_chunk("data/company_handbook.md")
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)
    vector_store.add_chunks(chunks, embeddings)

    llm = MockLLMClient()
    session_mgr = SessionManager()
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        llm=llm,
        session_manager=session_mgr,
    )

    session_id = "integration-session-42"

    # Turn 1: Ask about laptop policy
    resp1 = pipeline.query(
        "What are the laptop options?",
        session_id=session_id,
    )
    assert resp1.answer is not None
    assert resp1.retrieval_latency_ms >= 0
    assert resp1.total_latency_ms >= 0

    session = session_mgr.get_session(session_id)
    assert session is not None
    assert len(session.history) == 2
    assert session.history[0].content == "What are the laptop options?"

    # Turn 2: Follow-up question referencing turn 1
    resp2 = pipeline.query(
        "And what about the home office stipend?",
        session_id=session_id,
    )
    assert resp2.answer is not None
    assert len(session.history) == 4
    assert session.history[2].content == "And what about the home office stipend?"


if __name__ == "__main__":
    test_conversation_session()
    test_query_reformulator()
    test_session_manager()
    test_rag_pipeline_multi_turn()
    print("\nALL CONVERSATIONAL MEMORY TESTS PASSED! [OK]\n")
