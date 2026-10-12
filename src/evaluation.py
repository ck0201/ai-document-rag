"""
Automated RAG Evaluation Suite (The RAG Triad).

Calculates objective metrics for:
1. Context Relevance: Ratio of retrieved text that is relevant to the question.
2. Faithfulness / Groundedness: Ratio of claims in the answer supported by the retrieved context.
3. Answer Relevance: Semantic alignment between user question and final answer.
"""

from dataclasses import dataclass
import re
from typing import List, Tuple
from src.embeddings import BaseEmbedder, cosine_similarity
from src.vector_store import SearchResult


@dataclass
class EvaluationScore:
    query: str
    context_relevance: float  # 0.0 - 1.0
    faithfulness: float        # 0.0 - 1.0
    answer_relevance: float    # 0.0 - 1.0
    passed: bool


class RAGEvaluator:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[SearchResult],
        min_threshold: float = 0.60,
    ) -> EvaluationScore:
        """Computes RAG Triad scores."""
        # 1. Context Relevance
        q_vec = self.embedder.embed_query(query)
        if not retrieved_chunks:
            context_rel = 0.0
        else:
            chunk_scores = [res.score for res in retrieved_chunks]
            # Average similarity score of retrieved chunks
            context_rel = float(sum(chunk_scores) / len(chunk_scores))

        # 2. Answer Relevance
        ans_vec = self.embedder.embed_query(answer)
        ans_rel = float(cosine_similarity(q_vec, ans_vec))

        # 3. Faithfulness (Groundedness)
        # Check claim sentences from answer against combined context text
        combined_context = " ".join(c.chunk.text for c in retrieved_chunks).lower()
        sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 15]

        if not sentences:
            faithfulness = 1.0
        else:
            supported = 0
            for sentence in sentences:
                words = [w.lower().strip(",;:\"'") for w in sentence.split() if len(w) > 3]
                # A sentence is supported if significant keywords appear in context
                matches = sum(1 for w in words if w in combined_context)
                if len(words) == 0 or (matches / len(words)) >= 0.40:
                    supported += 1
            faithfulness = float(supported / len(sentences))

        passed = (context_rel >= 0.40 and faithfulness >= 0.70 and ans_rel >= 0.40)

        return EvaluationScore(
            query=query,
            context_relevance=round(context_rel, 4),
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(ans_rel, 4),
            passed=passed,
        )
