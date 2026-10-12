"""
Advanced Hybrid Search & Cross-Encoder Re-Ranking Pipeline.

Combines:
1. Dense Semantic Search (pgvector / ChromaDB)
2. Sparse Keyword Search (BM25Okapi)
3. Reciprocal Rank Fusion (RRF)
4. Cross-Encoder Re-Ranking (FlashRank ms-marco-TinyBERT)
"""

from collections import defaultdict
import re
from typing import Any, Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

from src.chunker import Chunk
from src.embeddings import BaseEmbedder
from src.vector_store import BaseVectorStore, SearchResult


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 sparse index for exact keyword matching."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.index: Optional[BM25Okapi] = None

    def fit(self, chunks: List[Chunk]):
        self.chunks = chunks
        if not chunks:
            self.index = None
            return
        corpus = [tokenize_text(c.text) for c in chunks]
        self.index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if not self.index or not self.chunks:
            return []

        tokens = tokenize_text(query)
        scores = self.index.get_scores(tokens)

        # Pair chunks with scores
        scored_pairs = list(zip(self.chunks, scores))
        # Filter out zero-score items and sort descending
        non_zero = [p for p in scored_pairs if p[1] > 0.0]
        non_zero.sort(key=lambda x: x[1], reverse=True)

        return [SearchResult(chunk=c, score=float(score)) for c, score in non_zero[:top_k]]


class CrossEncoderReranker:
    """Deep Cross-Attention Re-Ranker using FlashRank."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
        if not candidates:
            return []

        passages = [
            {"id": i, "text": res.chunk.text, "meta": res.chunk}
            for i, res in enumerate(candidates)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_req)

        final_results: List[SearchResult] = []
        for r in ranked_results[:top_k]:
            chunk = r["meta"]
            final_results.append(SearchResult(chunk=chunk, score=float(r["score"])))

        return final_results


class HybridRetriever:
    """
    Unified Hybrid Retrieval Engine combining Dense + Sparse + Cross-Encoder.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_index = BM25Index()
        self.reranker = CrossEncoderReranker()
        self.rrf_k = rrf_k

    def update_bm25_corpus(self, chunks: List[Chunk]):
        """Updates the sparse keyword index with latest chunks."""
        self.bm25_index.fit(chunks)

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[SearchResult]:
        """
        Combines two ranked lists using Reciprocal Rank Fusion:
        RRF(d) = sum( weight / (k + rank) )
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, Chunk] = {}

        # 1. Process Dense Ranks
        for rank, res in enumerate(dense_results, 1):
            c_id = res.chunk.chunk_id
            chunk_map[c_id] = res.chunk
            rrf_scores[c_id] += dense_weight / (self.rrf_k + rank)

        # 2. Process Sparse (BM25) Ranks
        for rank, res in enumerate(sparse_results, 1):
            c_id = res.chunk.chunk_id
            chunk_map[c_id] = res.chunk
            rrf_scores[c_id] += sparse_weight / (self.rrf_k + rank)

        # 3. Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [
            SearchResult(chunk=chunk_map[c_id], score=rrf_scores[c_id])
            for c_id in sorted_ids
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        use_reranker: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Full 3-stage retrieval pipeline:
        Stage 1: Fetch Dense Candidates (Top-10)
        Stage 2: Fetch Sparse BM25 Candidates (Top-10)
        Stage 3: Fuse via RRF -> Re-Rank with Cross-Encoder -> Top-K
        """
        # Stage 1: Dense Semantic Search
        q_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(q_vec, top_k=10, filter_metadata=filter_metadata)

        # Stage 2: Sparse Keyword Search (BM25)
        sparse_results = self.bm25_index.search(query, top_k=10)

        # Stage 3: Merge via Reciprocal Rank Fusion
        fused_candidates = self._reciprocal_rank_fusion(dense_results, sparse_results)

        if not use_reranker:
            return fused_candidates[:top_k]

        # Stage 4: Cross-Encoder Re-Ranking
        top_reranked = self.reranker.rerank(query, fused_candidates[:10], top_k=top_k)
        return top_reranked
