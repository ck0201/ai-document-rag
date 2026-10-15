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

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.jobs import IngestionJob, JobManager, JobStatus
from src.rag_pipeline import RAGPipeline
from src.guardrails import SecurityGuardrails
from src.rate_limiter import limiter


# Storage and Rate Limiting Constants
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024       # 2 MB per document
MAX_USER_STORAGE_BYTES = 10 * 1024 * 1024  # 10 MB cumulative per user


# Global state
job_manager = JobManager()
rag_pipeline: Optional[RAGPipeline] = None
guardrails = SecurityGuardrails()
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def background_ingest_worker(job_id: str, file_path: Path, source_name: str, user_id: Optional[str] = None):
    """Background worker function executed off the main request thread."""
    global rag_pipeline
    job_manager.update_status(job_id, JobStatus.PROCESSING)
    try:
        if rag_pipeline is None:
            rag_pipeline = RAGPipeline()

        chunks_indexed = rag_pipeline.ingest_document(file_path, source_name=source_name, user_id=user_id)
        job_manager.update_status(job_id, JobStatus.INDEXED, chunks_count=chunks_indexed)
    except Exception as exc:
        job_manager.update_status(job_id, JobStatus.FAILED, error_message=str(exc))
        # Purge temporary file if ingestion fails
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass


def cleanup_orphaned_uploads():
    """Sweeps data/uploads and removes files that do not belong to active indexed documents."""
    if not UPLOAD_DIR.exists() or rag_pipeline is None:
        return
    try:
        docs = rag_pipeline.vector_store.list_documents()
        active_sources = {d["source"] for d in docs if "source" in d}
    except Exception:
        return

    for f in UPLOAD_DIR.iterdir():
        if f.name == ".gitkeep" or f.is_dir():
            continue
        # An upload is active if its filename ends with an active source
        is_active = any(f.name.endswith(f"_{src}") or f.name == src for src in active_sources)
        if not is_active:
            try:
                f.unlink()
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global rag_pipeline
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    rag_pipeline = RAGPipeline()
    cleanup_orphaned_uploads()
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def add_no_cache_for_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serves the interactive web UI studio."""
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    )


# =====================================================================
# Request & Response Schemas
# =====================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question", min_length=2)
    top_k: int = Field(4, description="Number of context chunks to retrieve", ge=1, le=10)
    min_similarity: float = Field(0.35, description="Minimum cosine similarity cutoff", ge=0.0, le=1.0)
    session_id: Optional[str] = Field(None, description="Optional conversational session ID for multi-turn history")
    user_id: Optional[str] = Field(None, description="Optional tenant/user ID for multi-user document isolation")


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
    user_id: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    filename: str
    user_id: Optional[str] = None
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
    user_id: Optional[str] = None
    storage_used_bytes: int = 0
    storage_limit_bytes: int = MAX_USER_STORAGE_BYTES
    storage_used_mb: float = 0.0
    storage_limit_mb: float = 10.0


# =====================================================================
# REST Endpoints
# =====================================================================

@app.get("/health", response_model=SystemStatsResponse, tags=["System"])
async def health_check(
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Service health, database status, and user storage quota."""
    effective_user = user_id or x_user_id
    count = rag_pipeline.vector_store.count(user_id=effective_user) if rag_pipeline else 0
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

    storage_used = 0
    if rag_pipeline and effective_user and hasattr(rag_pipeline.vector_store, "get_user_storage_bytes"):
        storage_used = rag_pipeline.vector_store.get_user_storage_bytes(effective_user)

    return SystemStatsResponse(
        status="healthy",
        total_indexed_chunks=count,
        vector_store_type=store_type,
        llm_provider=llm_name,
        user_id=effective_user,
        storage_used_bytes=storage_used,
        storage_limit_bytes=MAX_USER_STORAGE_BYTES,
        storage_used_mb=round(storage_used / (1024 * 1024), 2),
        storage_limit_mb=round(MAX_USER_STORAGE_BYTES / (1024 * 1024), 2),
    )


@app.post(
    "/api/documents/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    tags=["Ingestion"],
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Asynchronous document upload endpoint with multi-tenant user isolation.
    Enforces 2 MB per document limit, 10 MB total user quota, and rate limits.
    """
    effective_user = user_id or x_user_id

    # 1. Enforce upload rate limit (max 5 uploads per minute)
    limiter.enforce(request, action="upload", max_requests=5, window_seconds=60, user_id=effective_user)

    # 2. Check allowed file extensions
    allowed_extensions = {".md", ".txt", ".pdf"}
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # 3. Check individual file size (2 MB limit)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed limit of 2 MB (uploaded: {file_size / (1024 * 1024):.2f} MB).",
        )

    # 4. Check total cumulative user workspace storage quota (10 MB limit)
    if effective_user and rag_pipeline and hasattr(rag_pipeline.vector_store, "get_user_storage_bytes"):
        current_storage = rag_pipeline.vector_store.get_user_storage_bytes(effective_user)
        if current_storage + file_size > MAX_USER_STORAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"User workspace storage quota of 10 MB exceeded. "
                    f"Current usage: {current_storage / (1024 * 1024):.2f} MB, "
                    f"file: {file_size / (1024 * 1024):.2f} MB. "
                    f"Please delete existing documents to free up space."
                ),
            )

    # Save to disk with user namespace to prevent cross-user file collisions
    safe_filename = Path(file.filename or "upload.txt").name
    user_prefix = effective_user if effective_user else "shared"
    dest_path = UPLOAD_DIR / f"{user_prefix}_{int(time.time())}_{safe_filename}"
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Register job and spawn async task
    job = job_manager.create_job(filename=safe_filename, file_path=str(dest_path), user_id=effective_user)
    background_tasks.add_task(background_ingest_worker, job.job_id, dest_path, safe_filename, effective_user)

    return JobResponse(
        job_id=job.job_id,
        filename=job.filename,
        user_id=job.user_id,
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
async def query_documents(req: QueryRequest, request: Request):
    """
    Queries the RAG pipeline with multi-turn conversation memory and tenant isolation.
    Embeds query -> Retrieves Top-K chunks -> Enforces Grounding -> Returns Answer + Citations.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")

    # Enforce free-tier query rate limit (max 12 requests per minute)
    limiter.enforce(request, action="query", max_requests=12, window_seconds=60, user_id=req.user_id)

    guard_res = guardrails.check_and_sanitize(req.query)
    if not guard_res.is_safe:
        raise HTTPException(status_code=400, detail=guard_res.rejection_reason)

    resp = rag_pipeline.query(
        question=guard_res.sanitized_text,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
        session_id=req.session_id,
        user_id=req.user_id,
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
        user_id=req.user_id,
    )


@app.post(
    "/api/query/stream",
    tags=["Retrieval"],
)
async def query_documents_stream(req: QueryRequest, request: Request):
    """
    Streams LLM output in real-time via Server-Sent Events (SSE) with tenant isolation.
    Yields 'sources' event with citations first, followed by 'token' events,
    and a final 'done' event with latency telemetry.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")

    # Enforce free-tier query rate limit (max 12 requests per minute)
    limiter.enforce(request, action="query", max_requests=12, window_seconds=60, user_id=req.user_id)

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
            user_id=req.user_id,
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
async def list_documents(
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Lists indexed documents and chunk counts for this user workspace."""
    effective_user = user_id or x_user_id
    if not rag_pipeline:
        return []
    if hasattr(rag_pipeline.vector_store, "list_documents"):
        return rag_pipeline.vector_store.list_documents(user_id=effective_user)
    return []


@app.get(
    "/api/suggestions",
    tags=["Retrieval"],
)
async def get_suggestions(
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Returns dynamic starter prompt cards tailored to the user's indexed documents."""
    effective_user = user_id or x_user_id
    if not rag_pipeline:
        return []
    return rag_pipeline.get_suggestions(user_id=effective_user)


@app.delete(
    "/api/documents/{source_name}",
    tags=["Ingestion"],
)
async def delete_document(
    source_name: str,
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Deletes all chunks associated with a document by filename for this user."""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")

    effective_user = user_id or x_user_id
    deleted_chunks = rag_pipeline.vector_store.delete_by_source(source_name, user_id=effective_user)
    rag_pipeline.clear_suggestions_cache()

    # Clean up physical files from uploads directory
    patterns = [f"*_{source_name}"]
    if effective_user:
        patterns.insert(0, f"{effective_user}_*_{source_name}")

    for pattern in patterns:
        for f in UPLOAD_DIR.glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass

    return {
        "status": "deleted",
        "source": source_name,
        "user_id": effective_user,
        "chunks_deleted": deleted_chunks,
        "remaining_chunks": rag_pipeline.vector_store.count(user_id=effective_user),
    }
