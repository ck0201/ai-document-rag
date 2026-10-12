"""
Test and Benchmark Suite for Hybrid Search (BM25 + Dense + RRF + Cross-Encoder)
and Multi-turn Conversational Memory.
"""

from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import DocumentParser
from src.chunker import RecursiveCharacterChunker
from src.embeddings import FastEmbedLocal
from src.vector_store import ChromaVectorStore
from src.hybrid_retriever import HybridRetriever
from src.memory import ConversationSession, QueryReformulator


def test_hybrid_and_reranking():
    print("=" * 70)
    print("PHASE 3: HYBRID SEARCH & CROSS-ENCODER RE-RANKING TESTS")
    print("=" * 70)

    # 1. Parse document
    chunker = RecursiveCharacterChunker(chunk_size=700, chunk_overlap=150)
    parser = DocumentParser(chunker)
    chunks = parser.parse_and_chunk("data/company_handbook.md")

    # 2. Embed & Index into local store
    embedder = FastEmbedLocal()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    store = ChromaVectorStore(collection_name="test_hybrid")
    store.add_chunks(chunks, embeddings)

    # 3. Initialize Hybrid Retriever
    retriever = HybridRetriever(vector_store=store, embedder=embedder)
    retriever.update_bm25_corpus(chunks)

    # 4. Test exact keyword query where BM25 excels
    keyword_query = "0.05% error rate"
    print(f"\n[+] Testing Keyword Query: '{keyword_query}'")
    bm25_matches = retriever.bm25_index.search(keyword_query, top_k=2)
    print(f"    BM25 Sparse Top Match: Chunk #{bm25_matches[0].chunk.metadata['chunk_index']} (Score: {bm25_matches[0].score:.4f})")
    assert bm25_matches[0].chunk.metadata["chunk_index"] == 3

    # 5. Test Full Hybrid + Cross-Encoder Re-Ranking Pipeline
    hybrid_query = "What happens if production error rate spikes during deployment canary?"
    print(f"\n[+] Testing Hybrid + Re-Ranking: '{hybrid_query}'")
    results = retriever.retrieve(hybrid_query, top_k=2, use_reranker=True)

    print(f"    Retrieved {len(results)} chunks after Cross-Encoder Re-ranking:")
    for rank, res in enumerate(results, 1):
        print(f"    Rank #{rank} | Reranker Score: {res.score:.4f} | Chunk #{res.chunk.metadata['chunk_index']}")
        print(f"    Preview: {res.chunk.text[:120].replace(chr(10), ' ')}...\n")

    assert len(results) == 2
    assert results[0].chunk.metadata["chunk_index"] == 3

    # 6. Test Multi-Turn Conversational Memory & Reformulation
    print("=" * 70)
    print("TESTING CONVERSATIONAL MEMORY & QUERY REFORMULATION")
    print("=" * 70)
    session = ConversationSession(session_id="session_test_1")
    session.add_user_message("What is the canary deployment policy?")
    session.add_assistant_message("Deployments must run canary analysis for at least 15 minutes...")

    reformulator = QueryReformulator()
    follow_up = "How long does it run?"
    reformulated = reformulator.reformulate(follow_up, session)
    print(f"\nOriginal follow-up:   '{follow_up}'")
    print(f"Reformulated query:    '{reformulated}'")
    assert "canary deployment" in reformulated.lower()

    print("\n" + "=" * 70)
    print("PHASE 3 ADVANCED RETRIEVAL TESTS PASSED 100%! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_hybrid_and_reranking()
