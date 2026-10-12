"""
Tests for ConversationalRouter and RAG conversational intent handling.
"""

from pathlib import Path
import sys
import json

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conversational import ConversationalRouter
from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaVectorStore
from src.embeddings import FastEmbedLocal
from src.llm import MockLLMClient


def test_conversational_router():
    print("=" * 70)
    print("TESTING CONVERSATIONAL ROUTER")
    print("=" * 70)

    # 1. Greetings
    greetings = ["hi", "Hello", "HEY!", "Good morning", "howdy there"]
    for g in greetings:
        is_conv, resp, _ = ConversationalRouter.classify_and_respond(g)
        assert is_conv is True, f"Failed to classify greeting: {g}"
        assert resp is not None and "Hello!" in resp
        print(f"  [GREETING] '{g}' -> {resp[:40]}...")

    # 2. Gratitude
    thanks = ["thank you", "thanks!", "thanks a lot", "ty", "perfect thank you"]
    for t in thanks:
        is_conv, resp, _ = ConversationalRouter.classify_and_respond(t)
        assert is_conv is True, f"Failed to classify gratitude: {t}"
        assert resp is not None and "welcome" in resp.lower()
        print(f"  [GRATITUDE] '{t}' -> {resp[:40]}...")

    # 3. Well-being
    wellbeing = ["how are you?", "how's it going", "how are you"]
    for w in wellbeing:
        is_conv, resp, _ = ConversationalRouter.classify_and_respond(w)
        assert is_conv is True, f"Failed to classify wellbeing: {w}"
        assert resp is not None and "doing well" in resp.lower()
        print(f"  [WELLBEING] '{w}' -> {resp[:40]}...")

    # 4. Identity
    identity = ["who are you", "what can you do", "help"]
    for i in identity:
        is_conv, resp, _ = ConversationalRouter.classify_and_respond(i)
        assert is_conv is True, f"Failed to classify identity: {i}"
        assert resp is not None and "Document" in resp
        print(f"  [IDENTITY] '{i}' -> {resp[:40]}...")

    # 5. Hybrid queries (prefix stripping)
    hybrid_cases = [
        ("Hi, what laptop choices do engineers receive?", "what laptop choices do engineers receive?"),
        ("Hello! What is the home office stipend?", "What is the home office stipend?"),
        ("Good morning - who approves expense reports?", "who approves expense reports?"),
    ]
    for raw, expected in hybrid_cases:
        is_conv, resp, stripped = ConversationalRouter.classify_and_respond(raw)
        assert is_conv is False, f"Hybrid query was falsely flagged as pure conversational: {raw}"
        assert resp is None
        assert stripped.lower() == expected.lower(), f"Expected '{expected}', got '{stripped}'"
        print(f"  [HYBRID] '{raw}' -> Cleaned: '{stripped}'")

    print("\nCONVERSATIONAL ROUTER TESTS PASSED! [OK]\n")


def test_rag_conversational_integration():
    print("=" * 70)
    print("TESTING RAG PIPELINE CONVERSATIONAL INTEGRATION")
    print("=" * 70)

    # Use lightweight mock LLM
    embedder = FastEmbedLocal()
    vector_store = ChromaVectorStore(collection_name="test_conv_collection")
    llm = MockLLMClient()
    pipeline = RAGPipeline(embedder=embedder, vector_store=vector_store, llm=llm)

    # 1. Pure chitchat via query()
    res = pipeline.query("thank you")
    assert "welcome" in res.answer.lower()
    assert len(res.sources) == 0
    assert res.context_chunks_used == 0
    print(f"  [query()] 'thank you' -> Response: '{res.answer}' | Sources: {res.sources}")

    # 2. Pure chitchat via query_stream()
    events = list(pipeline.query_stream("hi"))
    sources_event = next(e for e in events if e["event"] == "sources")
    assert sources_event["data"] == []
    token_events = [e for e in events if e["event"] == "token"]
    full_text = "".join(e["data"]["token"] for e in token_events)
    assert "Hello!" in full_text
    done_event = next(e for e in events if e["event"] == "done")
    assert done_event["data"]["status"] == "complete"
    print(f"  [query_stream()] 'hi' -> Streamed text: '{full_text}' | Done event received.")

    print("\nRAG CONVERSATIONAL INTEGRATION TESTS PASSED! [OK]\n")


if __name__ == "__main__":
    test_conversational_router()
    test_rag_conversational_integration()
