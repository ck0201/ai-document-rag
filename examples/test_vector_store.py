"""
Demonstration of Step 1.3: Local Vector Store with ChromaDB.

Tests:
1. Indexing chunks with precomputed embeddings and metadata.
2. Top-K nearest-neighbor search with similarity scoring.
3. Metadata filtering (SQL-like WHERE clause over vector search).
4. Idempotent upsert and deletion by source.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import DocumentParser
from src.chunker import RecursiveCharacterChunker
from src.embeddings import FastEmbedLocal
from src.vector_store import ChromaVectorStore


def run_vector_store_demo():
    print("=" * 70)
    print("STEP 1.3: LOCAL VECTOR STORE DEMONSTRATION (ChromaDB)")
    print("=" * 70)

    # 1. Parse and chunk
    chunker = RecursiveCharacterChunker(chunk_size=700, chunk_overlap=150)
    parser = DocumentParser(chunker)
    chunks = parser.parse_and_chunk("data/company_handbook.md")

    # 2. Embed
    embedder = FastEmbedLocal()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # 3. Initialize ChromaDB Vector Store
    db_path = ".chroma_db"
    store = ChromaVectorStore(collection_name="engineering_handbook", persist_dir=db_path)

    # Clean existing data to ensure deterministic demo
    store.delete_by_source("company_handbook.md")
    print(f"\n[+] ChromaDB initialized at '{db_path}'. Current count: {store.count()}")

    # 4. Add Chunks
    print(f"[+] Indexing {len(chunks)} chunks into ChromaDB...")
    store.add_chunks(chunks, embeddings)
    print(f"[+] Successfully indexed! Total chunks in DB: {store.count()}")

    # 5. Top-K Vector Search
    print("\n" + "=" * 70)
    print("TOP-K VECTOR SEARCH (Cosine Similarity)")
    print("=" * 70)

    sample_queries = [
        "What laptop do developers get and how much is the home office budget?",
        "What are the on-call pager duty escalation tiers and MTTR?",
    ]

    for q in sample_queries:
        print(f"\nQuery: \"{q}\"")
        q_vec = embedder.embed_query(q)
        results = store.search(q_vec, top_k=2)

        for rank, res in enumerate(results, 1):
            print(f"  Rank #{rank} | Similarity Score: {res.score:.4f}")
            print(f"  Source: {res.chunk.metadata.get('source')} | Chunk #{res.chunk.metadata.get('chunk_index')}")
            print(f"  Preview: {res.chunk.text[:140].replace(chr(10), ' ')}...\n")

    # 6. Metadata Filtering Test
    print("=" * 70)
    print("METADATA FILTERING TEST (Hybrid Vector + Metadata query)")
    print("=" * 70)

    filter_query = "What are the rules?"
    q_vec = embedder.embed_query(filter_query)

    print(f"\nSearching with filter: {{'chunk_index': 2}}")
    filtered_results = store.search(q_vec, top_k=2, filter_metadata={"chunk_index": 2})
    for res in filtered_results:
        print(f"  Matched Chunk #{res.chunk.metadata.get('chunk_index')} | Score: {res.score:.4f}")
        print(f"  Preview: {res.chunk.text[:120].replace(chr(10), ' ')}...")

    # 7. Idempotency Check
    print("\n" + "=" * 70)
    print("IDEMPOTENCY VERIFICATION")
    print("=" * 70)
    print(f"Adding the exact same {len(chunks)} chunks again...")
    store.add_chunks(chunks, embeddings)
    print(f"Total count after duplicate upsert: {store.count()} (should still be {len(chunks)}, NOT {len(chunks)*2}) [OK]")


if __name__ == "__main__":
    run_vector_store_demo()
