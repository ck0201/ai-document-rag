"""
Integration test for PostgreSQL + pgvector vector store.
"""

from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import DocumentParser
from src.chunker import RecursiveCharacterChunker
from src.embeddings import FastEmbedLocal
from src.pgvector_store import PGVectorStore


def test_pgvector_integration():
    print("=" * 70)
    print("TESTING POSTGRESQL + PGVECTOR (HNSW INDEX & SQL RETRIEVAL)")
    print("=" * 70)

    # 1. Parse and chunk sample document
    chunker = RecursiveCharacterChunker(chunk_size=700, chunk_overlap=150)
    parser = DocumentParser(chunker)
    chunks = parser.parse_and_chunk("data/company_handbook.md")

    # 2. Embed
    embedder = FastEmbedLocal()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # 3. Connect to pgvector
    store = PGVectorStore(
        dbname="rag_db",
        user="postgres",
        password="password123",
        host="localhost",
        port=5432,
    )
    store.delete_by_source("company_handbook.md")
    print(f"\n[+] Connected to PostgreSQL. Initial count: {store.count()}")

    # 4. Add chunks
    print(f"[+] Indexing {len(chunks)} chunks into PostgreSQL document_chunks table...")
    store.add_chunks(chunks, embeddings)
    total_count = store.count()
    print(f"[+] Successfully stored! Total chunks in Postgres: {total_count}")
    assert total_count == len(chunks), f"Expected {len(chunks)}, got {total_count}"

    # 5. Semantic Search via SQL
    query = "What hardware keys and approvals are required for database access?"
    print(f"\n[+] Query: '{query}'")
    q_vec = embedder.embed_query(query)
    results = store.search(q_vec, top_k=2)

    for rank, res in enumerate(results, 1):
        print(f"  Rank #{rank} | Cosine Similarity: {res.score:.4f}")
        print(f"  Source: {res.chunk.metadata.get('source')} | Chunk #{res.chunk.metadata.get('chunk_index')}")
        print(f"  Preview: {res.chunk.text[:130].replace(chr(10), ' ')}...\n")

    assert len(results) == 2
    assert results[0].chunk.metadata.get("chunk_index") == 1  # Security & DB access chunk

    # 6. JSONB Metadata Filtering
    print("[+] Testing JSONB Metadata Filtering (chunk_index = 2)...")
    filtered = store.search(q_vec, top_k=2, filter_metadata={"chunk_index": 2})
    assert len(filtered) == 1
    assert filtered[0].chunk.metadata.get("chunk_index") == 2
    print(f"  Matched Chunk #{filtered[0].chunk.metadata.get('chunk_index')} | Score: {filtered[0].score:.4f} [OK]")

    # 7. Idempotency Check
    print("[+] Testing Idempotency (re-inserting same chunks)...")
    store.add_chunks(chunks, embeddings)
    assert store.count() == len(chunks), "Upsert failed: duplicate rows created"
    print(f"  Total rows after duplicate upsert: {store.count()} [OK]")

    print("\n" + "=" * 70)
    print("PGVECTOR INTEGRATION TESTS PASSED 100%! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_pgvector_integration()
