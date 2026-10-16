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

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
        min_relative_score: float = 0.12,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        orig_scores = {res.chunk.chunk_id: res.score for res in candidates}

        passages = [
            {"id": i, "text": res.chunk.text, "meta": res.chunk}
            for i, res in enumerate(candidates)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_req)
        if not ranked_results:
            return candidates[:top_k]

        top_score = float(ranked_results[0].get("score", 0.0))

        final_results: List[SearchResult] = []
        for r in ranked_results:
            r_score = float(r.get("score", 0.0))
            # Token efficiency: Prune irrelevant tail noise when confidence drops sharply
            if top_score > 0 and (r_score / top_score) < min_relative_score and len(final_results) >= 2:
                break

            chunk = r["meta"]
            sim = orig_scores.get(chunk.chunk_id, float(r_score))
            final_results.append(SearchResult(chunk=chunk, score=float(sim)))
            if len(final_results) >= top_k:
                break

        return final_results if final_results else [
            SearchResult(chunk=ranked_results[0]["meta"], score=float(orig_scores.get(ranked_results[0]["meta"].chunk_id, 0.5)))
        ]


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
        self.bm25_indices: Dict[str, BM25Index] = {}
        self.reranker = CrossEncoderReranker()
        self.rrf_k = rrf_k

    def _get_or_build_bm25(self, user_id: Optional[str] = None) -> BM25Index:
        """Retrieves or lazily builds the BM25 index for the tenant."""
        cache_key = user_id or "global"
        if cache_key not in self.bm25_indices:
            idx = BM25Index()
            try:
                chunks = self.vector_store.get_all_chunks(user_id=user_id)
                idx.fit(chunks)
            except Exception:
                idx.fit([])
            self.bm25_indices[cache_key] = idx
        return self.bm25_indices[cache_key]

    def invalidate_bm25(self, user_id: Optional[str] = None):
        """Invalidates BM25 index when documents are added or removed."""
        if user_id:
            self.bm25_indices.pop(user_id, None)
            self.bm25_indices.pop("global", None)
        else:
            self.bm25_indices.clear()

    def update_bm25_corpus(self, chunks: List[Chunk], user_id: Optional[str] = None):
        """Manually updates the sparse keyword index with latest chunks."""
        cache_key = user_id or "global"
        idx = BM25Index()
        idx.fit(chunks)
        self.bm25_indices[cache_key] = idx

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
        Ranks by reciprocal score while preserving true similarity scores for thresholding.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, Chunk] = {}
        score_map: Dict[str, float] = {}

        # 1. Process Dense Ranks
        for rank, res in enumerate(dense_results, 1):
            c_id = res.chunk.chunk_id
            chunk_map[c_id] = res.chunk
            score_map[c_id] = max(score_map.get(c_id, 0.0), res.score)
            rrf_scores[c_id] += dense_weight / (self.rrf_k + rank)

        # 2. Process Sparse (BM25) Ranks
        max_sparse = max((r.score for r in sparse_results), default=1.0) or 1.0
        for rank, res in enumerate(sparse_results, 1):
            c_id = res.chunk.chunk_id
            chunk_map[c_id] = res.chunk
            norm_bm25 = min(0.95, max(0.40, res.score / max_sparse))
            score_map[c_id] = max(score_map.get(c_id, 0.0), norm_bm25)
            rrf_scores[c_id] += sparse_weight / (self.rrf_k + rank)

        # 3. Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [
            SearchResult(chunk=chunk_map[c_id], score=score_map.get(c_id, 0.50))
            for c_id in sorted_ids
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        use_reranker: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Full 3-stage retrieval pipeline:
        Stage 1: Fetch Dense Candidates (Pool of top 16)
        Stage 2: Fetch Sparse BM25 Candidates (Pool of top 16)
        Stage 3: Fuse via RRF
        Stage 4: Cross-Encoder Re-Ranking -> Top-K
        """
        pool_size = max(16, top_k * 2)

        # Stage 1: Dense Semantic Search
        q_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(q_vec, top_k=pool_size, filter_metadata=filter_metadata)

        # Stage 2: Sparse Keyword Search (BM25)
        bm25 = self._get_or_build_bm25(user_id)
        sparse_results = bm25.search(query, top_k=pool_size)

        # Stage 3: Merge via Reciprocal Rank Fusion
        fused_candidates = self._reciprocal_rank_fusion(dense_results, sparse_results)
        if not fused_candidates:
            return dense_results[:top_k]

        if not use_reranker:
            return fused_candidates[:top_k]

        # Stage 4: Cross-Encoder Re-Ranking
        try:
            candidates_to_rerank = fused_candidates[:pool_size]
            top_reranked = self.reranker.rerank(query, candidates_to_rerank, top_k=top_k)
            return top_reranked if top_reranked else fused_candidates[:top_k]
        except Exception:
            return fused_candidates[:top_k]
