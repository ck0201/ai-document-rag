"""
Embeddings & Vector Similarity Engine.

Demonstrates:
1. High-dimensional vector generation.
2. Vector normalization (L2 norm).
3. Cosine similarity vs dot product calculation with pure NumPy.
"""

from abc import ABC, abstractmethod
from typing import List, Sequence
import numpy as np


class BaseEmbedder(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embeds a list of texts into a 2D numpy array of shape (N, dimension).
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string into a 1D numpy array of shape (dimension,).
        """
        pass


class FastEmbedLocal(BaseEmbedder):
    """
    Local embedding using fastembed (ONNX runtime).
    Default model: 'BAAI/bge-small-en-v1.5' (384 dimensions, fast on CPU).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self.dimension = 384

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        # fastembed returns a generator of numpy arrays
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        # For bge models, queries can be embedded directly
        embedding = list(self.model.embed([query]))[0]
        return np.array(embedding, dtype=np.float32)


# =====================================================================
# Vector Similarity Math (NumPy implementation)
# =====================================================================

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes vectors to unit length (L2 norm = 1.0).
    If shape is 1D (D,) -> returns (D,)
    If shape is 2D (N, D) -> normalizes each row independently.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm > 0 else vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes standard cosine similarity between two 1D vectors:
    cos(theta) = (u . v) / (||u|| * ||v||)
    """
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(dot_product / (norm_u * norm_v))


def batch_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a 1D query vector and a 2D matrix of document vectors.
    Doc vectors: (N, D), Query: (D,) -> returns (N,) similarity scores.
    """
    norm_query = l2_normalize(query_vec)
    norm_docs = l2_normalize(doc_vecs)
    # When both are unit vectors, cosine similarity == matrix-vector dot product!
    return np.dot(norm_docs, norm_query)
