"""
Automated Integration Tests for RAG REST API.
"""

from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import app


def test_rag_rest_api():
    print("=" * 70)
    print("RUNNING FASTAPI REST API INTEGRATION TESTS")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Health Check
        print("\n[1] Testing GET /health...")
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Expected 200, got {health_resp.status_code}"
        data = health_resp.json()
        print(f"    Health Status: {data['status']}, Total Chunks: {data['total_indexed_chunks']}")

        # 2. Upload Document (Async 202 Accepted)
        print("\n[2] Testing POST /api/documents/upload (Async Ingestion)...")
        sample_doc = Path("data/company_handbook.md")
        with sample_doc.open("rb") as f:
            upload_resp = client.post(
                "/api/documents/upload",
                files={"file": (sample_doc.name, f, "text/markdown")},
            )

        assert upload_resp.status_code == 202, f"Expected 202, got {upload_resp.status_code}"
        job_data = upload_resp.json()
        job_id = job_data["job_id"]
        print(f"    Upload Accepted! Job ID: {job_id}, Initial Status: {job_data['status']}")

        # 3. Poll Job Status
        print(f"\n[3] Polling GET /api/documents/jobs/{job_id}...")
        final_status = None
        for _ in range(20):
            job_status_resp = client.get(f"/api/documents/jobs/{job_id}")
            assert job_status_resp.status_code == 200
            status_data = job_status_resp.json()
            final_status = status_data["status"]
            if final_status in ("INDEXED", "FAILED"):
                break
            time.sleep(0.5)

        print(f"    Job Reached Terminal State: {final_status} (Indexed Chunks: {status_data['chunks_count']})")
        assert final_status == "INDEXED", f"Expected job status INDEXED, got {final_status}"

        # 4. Query Documents
        print("\n[4] Testing POST /api/query...")
        query_payload = {
            "query": "What are the laptop options and home office stipend?",
            "top_k": 2,
            "min_similarity": 0.35,
        }
        query_resp = client.post("/api/query", json=query_payload)
        assert query_resp.status_code == 200, f"Expected 200, got {query_resp.status_code}"
        q_data = query_resp.json()

        print(f"    Query Latency: {q_data['latency_ms']:.2f} ms")
        print(f"    Sources Cited: {len(q_data['sources'])}")
        print(f"    Top Source: {q_data['sources'][0]['source']} (Score: {q_data['sources'][0]['similarity']})")
        print(f"    Answer Preview: {q_data['answer'][:120].replace(chr(10), ' ')}...")
        assert len(q_data["sources"]) > 0

        # 5. Unsupported file format rejection
        print("\n[5] Testing Validation (Reject unsupported file extensions)...")
        bad_upload = client.post(
            "/api/documents/upload",
            files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
        )
        assert bad_upload.status_code == 400
        print(f"    Rejected bad file extension with HTTP 400 as expected: {bad_upload.json()['detail']}")

        # 6. Delete document
        print("\n[6] Testing DELETE /api/documents/{source_name}...")
        del_resp = client.delete("/api/documents/company_handbook.md")
        assert del_resp.status_code == 200
        print(f"    Deletion Success: {del_resp.json()}")

    print("\n" + "=" * 70)
    print("ALL API INTEGRATION TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_rag_rest_api()
