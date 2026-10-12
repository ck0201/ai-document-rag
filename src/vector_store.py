"""
Vector Store Engine.

Provides an abstraction layer and ChromaDB implementation for:
1. Chunk indexing with vectors and rich metadata.
2. Top-K nearest neighbor retrieval.
3. Metadata filtering (e.g. filter by source document or page).
4. Idempotent upsert and document deletion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.chunker import Chunk


@dataclass
class SearchResult:
    """Represents a matched chunk with similarity score."""
    chunk: Chunk
    score: float

    def __repr__(self) -> str:
        return (
            f"SearchResult(score={self.score:.4f}, "
            f"source={self.chunk.metadata.get('source')}, "
            f"chunk_idx={self.chunk.metadata.get('chunk_index')}, "
            f"text='{self.chunk.text[:60]}...')"
        )


class BaseVectorStore(ABC):
    """Abstract interface for Vector Databases."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Indexes chunks with their precomputed embeddings."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Performs nearest-neighbor vector search."""
        pass

    @abstractmethod
    def delete_by_source(self, source_name: str) -> None:
        """Deletes all chunks belonging to a specific document (idempotency/cleanup)."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns total number of indexed chunks."""
        pass

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns list of distinct indexed documents and their chunk counts."""
        pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = PROJECT_ROOT / ".chroma_db"


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation with persistent local storage.
    Uses HNSW (Hierarchical Navigable Small World) index under the hood.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_dir: Optional[str | Path] = None,
    ):
        import chromadb
        from chromadb.config import Settings

        if persist_dir is None:
            self.persist_dir = DEFAULT_PERSIST_DIR
        else:
            p = Path(persist_dir)
            self.persist_dir = p if p.is_absolute() else (PROJECT_ROOT / p)

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        # Using cosine distance for vector comparison
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings_list = embeddings.tolist()

        # Chroma's upsert replaces chunks if id already exists (idempotency)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        # query_embedding shape can be (D,) -> convert to list of list
        query_vec = [query_embedding.tolist()] if query_embedding.ndim == 1 else query_embedding.tolist()

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": query_vec,
            "n_results": min(top_k, self.count()) if self.count() > 0 else top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self.collection.query(**query_kwargs)

        search_results: List[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine distance (0 to 2, where 0 is identical)
            # Similarity score = 1 - distance
            similarity = 1.0 - dist

            chunk = Chunk(
                chunk_id=chunk_id,
                text=doc,
                char_count=len(doc),
                metadata=meta,
            )
            search_results.append(SearchResult(chunk=chunk, score=similarity))

        return search_results

    def delete_by_source(self, source_name: str) -> None:
        self.collection.delete(where={"source": source_name})

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> List[Dict[str, Any]]:
        results = self.collection.get(include=["metadatas"])
        counts: Dict[str, int] = {}
        if results and "metadatas" in results and results["metadatas"]:
            for m in results["metadatas"]:
                if m and isinstance(m, dict):
                    src = m.get("source", "unknown")
                    counts[src] = counts.get(src, 0) + 1
        return [{"source": k, "chunks_count": v} for k, v in sorted(counts.items())]
