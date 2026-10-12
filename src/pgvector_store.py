"""
Enterprise PostgreSQL Vector Store using pgvector.

Features:
- Native pgvector extension with vector(384) column.
- HNSW (Hierarchical Navigable Small World) index for sub-millisecond similarity queries.
- Cosine distance operator (<=>) transformed into similarity score.
- JSONB metadata storage with GIN indexing for fast filtering.
- Idempotent upserts via ON CONFLICT (id) DO UPDATE.
"""

import json
from typing import Any, Dict, List, Optional
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pgvector.psycopg2 import register_vector

from src.chunker import Chunk
from src.vector_store import BaseVectorStore, SearchResult


class PGVectorStore(BaseVectorStore):
    def __init__(
        self,
        dbname: str = "rag_db",
        user: str = "postgres",
        password: str = "password123",
        host: str = "localhost",
        port: int = 5432,
        embedding_dim: int = 384,
    ):
        self.conn_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port,
        }
        self.embedding_dim = embedding_dim
        self._init_db()

    def _get_connection(self):
        conn = psycopg2.connect(**self.conn_params)
        register_vector(conn)
        return conn

    def _init_db(self):
        """Initializes pgvector extension and relational chunks table with HNSW index."""
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                # 1. Enable extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # 2. Create chunks table
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id VARCHAR(64) PRIMARY KEY,
                    source VARCHAR(255) NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.embedding_dim}) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)

                # 3. Create HNSW index for cosine distance (<=>)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_hnsw 
                ON document_chunks 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """)

                # 4. Create metadata B-tree and GIN indexes
                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON document_chunks (source);
                CREATE INDEX IF NOT EXISTS idx_chunks_meta ON document_chunks USING gin (metadata);
                """)
            conn.commit()

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return

        records = []
        for c, emb in zip(chunks, embeddings):
            meta_json = json.dumps(c.metadata)
            records.append((
                c.chunk_id,
                c.metadata.get("source", "unknown"),
                c.text,
                meta_json,
                emb,
            ))

        query = """
        INSERT INTO document_chunks (id, source, text, metadata, embedding)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            source = EXCLUDED.source,
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding;
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    query,
                    records,
                    template="(%s, %s, %s, %s, %s)",
                )
            conn.commit()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        where_clauses = ["1=1"]
        params: List[Any] = [query_embedding]

        if filter_metadata:
            for k, v in filter_metadata.items():
                if k == "source":
                    where_clauses.append("source = %s")
                    params.append(v)
                else:
                    where_clauses.append("metadata @> %s::jsonb")
                    params.append(json.dumps({k: v}))

        where_sql = " AND ".join(where_clauses)
        params.extend([query_embedding, top_k])

        sql = f"""
        SELECT 
            id,
            source,
            text,
            metadata,
            1.0 - (embedding <=> %s) AS similarity
        FROM document_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s ASC
        LIMIT %s;
        """

        results: List[SearchResult] = []
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                for row in rows:
                    meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
                    chunk = Chunk(
                        chunk_id=row["id"],
                        text=row["text"],
                        char_count=len(row["text"]),
                        metadata=meta,
                    )
                    results.append(SearchResult(chunk=chunk, score=float(row["similarity"])))

        return results

    def delete_by_source(self, source_name: str) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE source = %s;", (source_name,))
            conn.commit()

    def count(self) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_chunks;")
                res = cur.fetchone()
                return int(res[0]) if res else 0

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT source, COUNT(*) FROM document_chunks GROUP BY source ORDER BY source;")
                rows = cur.fetchall()
                return [{"source": row[0], "chunks_count": int(row[1])} for row in rows]
