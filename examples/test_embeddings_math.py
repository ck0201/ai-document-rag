"""
Demonstration of Step 1.2: Embeddings & Vector Similarity Math.

Tests:
1. High-dimensional vector generation for real document chunks.
2. Cosine similarity vs Dot product.
3. Verification of why L2-normalized dot product equals cosine similarity.
4. Semantic search ranking on sample queries.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import DocumentParser
from src.chunker import RecursiveCharacterChunker
from src.embeddings import FastEmbedLocal, cosine_similarity, batch_cosine_similarity, l2_normalize


def run_embedding_demo():
    print("=" * 70)
    print("STEP 1.2: EMBEDDINGS & VECTOR SIMILARITY DEMONSTRATION")
    print("=" * 70)

    # 1. Parse and chunk our sample document
    chunker = RecursiveCharacterChunker(chunk_size=700, chunk_overlap=150)
    parser = DocumentParser(chunker)
    chunks = parser.parse_and_chunk("data/company_handbook.md")
    print(f"\n[+] Loaded and chunked document into {len(chunks)} chunks.")

    # 2. Initialize Embedder
    print("[+] Initializing FastEmbed embedding model (BAAI/bge-small-en-v1.5)...")
    embedder = FastEmbedLocal()

    # 3. Generate embeddings
    chunk_texts = [c.text for c in chunks]
    print(f"[+] Generating embeddings for {len(chunk_texts)} chunks...")
    doc_embeddings = embedder.embed_texts(chunk_texts)

    print(f"    Vector Matrix Shape: {doc_embeddings.shape}")
    print(f"    (Rows = {doc_embeddings.shape[0]} chunks, Columns = {doc_embeddings.shape[1]} vector dimensions)")

    # 4. Math Inspection: Inspect Vector Norms and Normalization
    raw_norm = np.linalg.norm(doc_embeddings[0])
    normalized_vecs = l2_normalize(doc_embeddings)
    normalized_norm = np.linalg.norm(normalized_vecs[0])

    print("\n--- VECTOR NORMALIZATION MATH ---")
    print(f"Raw Vector L2 Norm (magnitude):        {raw_norm:.4f}")
    print(f"Normalized Vector L2 Norm:            {normalized_norm:.4f}  (Unit vector length = 1.0)")

    # 5. Queries to test semantic matching
    queries = [
        "What laptop and home office stipend do engineers receive?",
        "How do engineers authenticate into production and access databases?",
        "What is the on-call response time and MTTR for critical Sev-1 outages?",
        "What is required before deploying code to production?",
    ]

    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY SEARCH ON REAL QUERIES")
    print("=" * 70)

    for q in queries:
        print(f"\nQuery: \"{q}\"")
        q_vec = embedder.embed_query(q)

        # Method A: Manual pairwise cosine similarity formula
        manual_scores = [cosine_similarity(q_vec, d_vec) for d_vec in doc_embeddings]

        # Method B: Matrix multiplication on normalized vectors (Dot product)
        q_norm = l2_normalize(q_vec)
        dot_product_scores = np.dot(normalized_vecs, q_norm)

        # Verify equivalence
        np.testing.assert_allclose(manual_scores, dot_product_scores, atol=1e-5)

        # Rank chunks by score descending
        ranked_indices = np.argsort(dot_product_scores)[::-1]

        top_idx = ranked_indices[0]
        top_score = dot_product_scores[top_idx]
        top_chunk = chunks[top_idx]

        print(f"  -> Best Match: Chunk #{top_chunk.metadata['chunk_index']} (Score: {top_score:.4f})")
        print(f"  -> Equivalence Check: Manual Cosine ({manual_scores[top_idx]:.4f}) == Normalized Dot Product ({dot_product_scores[top_idx]:.4f}) [OK]")
        print(f"  -> Content Preview:\n     {top_chunk.text[:220].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    run_embedding_demo()
