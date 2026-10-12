"""
Production FastAPI Service for Document Q&A RAG Bot.

Features:
- Non-blocking async document uploads with BackgroundTasks.
- Ingestion job status tracking (PENDING -> PROCESSING -> INDEXED).
- Low-latency query endpoint with source citations & latency metrics.
- Vector database management and health checks.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.jobs import IngestionJob, JobManager, JobStatus
from src.rag_pipeline import RAGPipeline
from src.guardrails import SecurityGuardrails


# Global state
job_manager = JobManager()
rag_pipeline: Optional[RAGPipeline] = None
guardrails = SecurityGuardrails()
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def background_ingest_worker(job_id: str, file_path: Path, source_name: str):
    """Background worker function executed off the main request thread."""
    global rag_pipeline
    job_manager.update_status(job_id, JobStatus.PROCESSING)
    try:
        if rag_pipeline is None:
            rag_pipeline = RAGPipeline()

        chunks_indexed = rag_pipeline.ingest_document(file_path, source_name=source_name)
        job_manager.update_status(job_id, JobStatus.INDEXED, chunks_count=chunks_indexed)
    except Exception as exc:
        job_manager.update_status(job_id, JobStatus.FAILED, error_message=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global rag_pipeline
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    rag_pipeline = RAGPipeline()
    yield


app = FastAPI(
    title="Enterprise AI Document Q&A (RAG API)",
    description="Production-ready REST API for asynchronous document indexing and grounded LLM retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serves the interactive web UI studio."""
    return FileResponse(STATIC_DIR / "index.html")


# =====================================================================
# Request & Response Schemas
# =====================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question", min_length=2)
    top_k: int = Field(4, description="Number of context chunks to retrieve", ge=1, le=10)
    min_similarity: float = Field(0.35, description="Minimum cosine similarity cutoff", ge=0.0, le=1.0)
    session_id: Optional[str] = Field(None, description="Optional conversational session ID for multi-turn history")


class SourceCitation(BaseModel):
    source: Optional[str] = None
    page: int = 1
    chunk_index: Optional[int] = None
    similarity: float = 0.0
    text: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class LatencyBreakdown(BaseModel):
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    chunks_used: int
    latency_ms: float
    latency_breakdown: Optional[LatencyBreakdown] = None
    groundedness: float = 1.0
    vector_store_type: str = "pgvector (PostgreSQL HNSW)"
    session_id: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    chunks_count: int
    error_message: Optional[str]
    created_at: str
    completed_at: Optional[str]


class SystemStatsResponse(BaseModel):
    status: str
    total_indexed_chunks: int
    vector_store_type: str
    llm_provider: Optional[str] = None


# =====================================================================
# REST Endpoints
# =====================================================================

@app.get("/health", response_model=SystemStatsResponse, tags=["System"])
async def health_check():
    """Service health and database status."""
    count = rag_pipeline.vector_store.count() if rag_pipeline else 0
    llm_name = "Unknown"
    if rag_pipeline and rag_pipeline.llm:
        from src.llm import GeminiLLMClient, OllamaLLMClient
        if isinstance(rag_pipeline.llm, GeminiLLMClient):
            llm_name = f"Gemini ({rag_pipeline.llm.model})"
        elif isinstance(rag_pipeline.llm, OllamaLLMClient):
            llm_name = f"Ollama ({rag_pipeline.llm.model})"
        else:
            llm_name = "Mock LLM"

    store_type = "ChromaDB (HNSW)"
    if rag_pipeline and hasattr(rag_pipeline.vector_store, "__class__") and "PGVector" in rag_pipeline.vector_store.__class__.__name__:
        store_type = "pgvector (PostgreSQL HNSW)"

    return SystemStatsResponse(
        status="healthy",
        total_indexed_chunks=count,
        vector_store_type=store_type,
        llm_provider=llm_name,
    )


@app.post(
    "/api/documents/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    tags=["Ingestion"],
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Asynchronous document upload endpoint.
    Saves file to disk, queues background worker, and immediately returns 202 Accepted.
    """
    allowed_extensions = {".md", ".txt", ".pdf"}
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save to disk
    safe_filename = Path(file.filename or "upload.txt").name
    dest_path = UPLOAD_DIR / f"{int(time.time())}_{safe_filename}"
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Register job and spawn async task
    job = job_manager.create_job(filename=safe_filename, file_path=str(dest_path))
    background_tasks.add_task(background_ingest_worker, job.job_id, dest_path, safe_filename)

    return JobResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status,
        chunks_count=job.chunks_count,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@app.get(
    "/api/documents/jobs/{job_id}",
    response_model=JobResponse,
    tags=["Ingestion"],
)
async def get_job_status(job_id: str):
    """Polls ingestion progress for a specific upload job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return JobResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status,
        chunks_count=job.chunks_count,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@app.post(
    "/api/query",
    response_model=QueryResponse,
    tags=["Retrieval"],
)
async def query_documents(req: QueryRequest):
    """
    Queries the RAG pipeline with multi-turn conversation memory.
    Embeds query -> Retrieves Top-K chunks -> Enforces Grounding -> Returns Answer + Citations.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")

    guard_res = guardrails.check_and_sanitize(req.query)
    if not guard_res.is_safe:
        raise HTTPException(status_code=400, detail=guard_res.rejection_reason)

    resp = rag_pipeline.query(
        question=guard_res.sanitized_text,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
        session_id=req.session_id,
    )

    return QueryResponse(
        query=resp.query,
        answer=resp.answer,
        sources=[
            SourceCitation(
                source=s.get("source"),
                page=s.get("page", 1),
                chunk_index=s.get("chunk_index"),
                similarity=s.get("similarity", 0.0),
                text=s.get("text"),
                char_start=s.get("char_start"),
                char_end=s.get("char_end"),
            )
            for s in resp.sources
        ],
        chunks_used=resp.context_chunks_used,
        latency_ms=round(resp.total_latency_ms, 2),
        latency_breakdown=LatencyBreakdown(
            retrieval_ms=round(resp.retrieval_latency_ms, 2),
            generation_ms=round(resp.generation_latency_ms, 2),
            total_ms=round(resp.total_latency_ms, 2),
        ),
        groundedness=resp.groundedness_score,
        vector_store_type=resp.vector_store_type,
        session_id=req.session_id,
    )


@app.post(
    "/api/query/stream",
    tags=["Retrieval"],
)
async def query_documents_stream(req: QueryRequest):
    """
    Streams LLM output in real-time via Server-Sent Events (SSE).
    Yields 'sources' event with citations first, followed by 'token' events,
    and a final 'done' event with latency telemetry.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")

    from fastapi.responses import StreamingResponse
    import json

    guard_res = guardrails.check_and_sanitize(req.query)
    if not guard_res.is_safe:
        def error_generator():
            err_data = json.dumps({"message": guard_res.rejection_reason})
            yield f"event: error\ndata: {err_data}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(error_generator(), media_type="text/event-stream")

    def sse_generator():
        for event in rag_pipeline.query_stream(
            question=guard_res.sanitized_text,
            top_k=req.top_k,
            min_similarity=req.min_similarity,
            session_id=req.session_id,
        ):
            event_name = event.get("event", "message")
            data = event.get("data", "")
            data_str = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
            yield f"event: {event_name}\ndata: {data_str}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete(
    "/api/sessions/{session_id}",
    tags=["Conversational Memory"],
)
async def delete_session(session_id: str):
    """Clears conversational memory history for a session."""
    if not rag_pipeline or not rag_pipeline.session_manager:
        raise HTTPException(status_code=500, detail="RAG Pipeline or SessionManager is not initialized")
    deleted = rag_pipeline.session_manager.delete_session(session_id)
    return {"status": "cleared" if deleted else "not_found", "session_id": session_id}


@app.get(
    "/api/documents",
    tags=["Ingestion"],
)
async def list_documents():
    """Lists all indexed documents and chunk counts from the active vector store."""
    if not rag_pipeline:
        return []
    if hasattr(rag_pipeline.vector_store, "list_documents"):
        return rag_pipeline.vector_store.list_documents()
    return []


@app.get(
    "/api/suggestions",
    tags=["Retrieval"],
)
async def get_suggestions():
    """Returns dynamic starter prompt cards tailored to the indexed documents."""
    if not rag_pipeline:
        return []
    return rag_pipeline.get_suggestions()


@app.delete(
    "/api/documents/{source_name}",
    tags=["Ingestion"],
)
async def delete_document(source_name: str):
    """Deletes all chunks associated with a document by filename."""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")

    rag_pipeline.vector_store.delete_by_source(source_name)
    rag_pipeline.clear_suggestions_cache()
    return {
        "status": "deleted",
        "source": source_name,
        "remaining_chunks": rag_pipeline.vector_store.count(),
    }
