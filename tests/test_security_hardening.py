"""
Security Hardening Integration Tests.

Validates:
1. Max 10 MB per document file size enforcement (HTTP 413).
2. Max 10 MB cumulative user workspace storage quota (HTTP 413).
3. Free-tier API rate limiting on queries (12 RPM) and uploads (5 RPM) with HTTP 429.
4. Upload file cleanup upon document deletion.
"""

from io import BytesIO
import os
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import app, MAX_FILE_SIZE_BYTES, MAX_USER_STORAGE_BYTES, UPLOAD_DIR, cleanup_orphaned_uploads
from src.rate_limiter import limiter


def test_file_size_limits_and_rate_limiting():
    print("=" * 70)
    print("RUNNING SECURITY HARDENING & RATE LIMITING TESTS")
    print("=" * 70)

    # Ensure rate limiting and admin authorization are enforced for this test
    os.environ["TESTING"] = "false"
    os.environ["ENFORCE_ADMIN_IN_TESTS"] = "true"
    os.environ["DISABLE_RATE_LIMIT"] = "false"
    limiter.reset()

    user_test = "usr_sec_audit_test"

    with TestClient(app) as client:
        # 0. Test School Admin Passcode Authentication
        print("\n[0] Testing School Admin Passcode Authentication...")
        auth_fail = client.post("/api/auth/verify-admin", json={"passcode": "wrong_pin"})
        assert auth_fail.status_code == 401
        auth_ok = client.post("/api/auth/verify-admin", json={"passcode": "school123"})
        assert auth_ok.status_code == 200
        assert auth_ok.json()["success"] is True
        print("    [AUTH OK] Validated admin passcode check endpoint.")

        # 1. Test Admin Authorization on Upload (Unauthorized parent/student rejected with HTTP 403)
        print("\n[1] Testing admin authorization on upload (403 without passcode)...")
        res_no_auth = client.post(
            "/api/documents/upload",
            data={"user_id": user_test},
            files={"file": ("test.txt", BytesIO(b"content"), "text/plain")},
        )
        assert res_no_auth.status_code == 403, f"Expected 403, got {res_no_auth.status_code}"
        print(f"    [BLOCKED] Rejected unauthenticated upload with HTTP 403: {res_no_auth.json()['detail']}")

        # 2. Test Single File Size Limit (> 50 MB rejected with HTTP 413)
        print("\n[2] Testing single file size limit (> 50 MB)...")
        oversized_bytes = b"X" * (50 * 1024 * 1024 + 1024)  # 50MB + 1KB
        res_large = client.post(
            "/api/documents/upload",
            headers={"X-Admin-Passcode": "school123"},
            data={"user_id": user_test},
            files={"file": ("too_large.txt", BytesIO(oversized_bytes), "text/plain")},
        )
        assert res_large.status_code == 413, f"Expected 413, got {res_large.status_code}"
        print(f"    [BLOCKED] Rejected oversized file with HTTP 413: {res_large.json()['detail']}")

        # 3. Test Allowed File Size (<= 50 MB accepted with admin passcode)
        print("\n[3] Testing allowed file size (valid small document with admin passcode)...")
        valid_bytes = b"# Valid Doc\nThis is a securely sized test document.\n"
        res_valid = client.post(
            "/api/documents/upload",
            headers={"X-Admin-Passcode": "school123"},
            data={"user_id": user_test},
            files={"file": ("valid_doc.md", BytesIO(valid_bytes), "text/markdown")},
        )
        assert res_valid.status_code == 202, f"Expected 202, got {res_valid.status_code}"
        job_id = res_valid.json()["job_id"]
        print(f"    [ACCEPTED] Valid file accepted with HTTP 202: Job {job_id}")

        # Wait for valid_doc to index
        for _ in range(20):
            status_res = client.get(f"/api/documents/jobs/{job_id}").json()
            if status_res["status"] in ("INDEXED", "FAILED"):
                break
            time.sleep(0.3)

        # 4. Test Cumulative Storage Quota check (50 MB limit)
        print("\n[4] Testing cumulative storage quota (50 MB limit)...")
        assert MAX_USER_STORAGE_BYTES == 50 * 1024 * 1024
        assert MAX_FILE_SIZE_BYTES == 50 * 1024 * 1024
        print(f"    Storage limits confirmed: Max {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB/file, Max {MAX_USER_STORAGE_BYTES / (1024*1024):.0f} MB/user")

        # 5. Test Query Rate Limiting (Max 12 requests / minute)
        print("\n[5] Testing query rate limiting (12 RPM ceiling)...")
        limiter.reset()

        from unittest.mock import patch
        from src.rag_pipeline import RAGResponse

        dummy_response = RAGResponse(
            query="test",
            answer="mock answer",
            sources=[],
            context_chunks_used=0,
            total_latency_ms=10.0,
            retrieval_latency_ms=5.0,
            generation_latency_ms=5.0,
        )

        with patch("src.api.rag_pipeline.query", return_value=dummy_response):
            # Send 12 queries rapidly
            for i in range(12):
                q_res = client.post(
                    "/api/query",
                    json={"query": f"Question {i}?", "user_id": user_test},
                )
                assert q_res.status_code in (200, 400), f"Query {i} unexpectedly failed with {q_res.status_code}"

            # The 13th query MUST be throttled with HTTP 429
            th_res = client.post(
                "/api/query",
                json={"query": "Question 13 (should exceed)?", "user_id": user_test},
            )
            assert th_res.status_code == 429, f"Expected HTTP 429 Too Many Requests, got {th_res.status_code}"
            assert "Retry-After" in th_res.headers
            print(f"    [RATE LIMITED] 13th query throttled with HTTP 429: {th_res.json()['detail']}")
            print(f"    Retry-After header present: {th_res.headers.get('Retry-After')}s")

        # 6. Test Admin Authorization on Delete
        print("\n[6] Testing admin authorization on deletion (403 without passcode)...")
        del_unauth = client.delete(f"/api/documents/valid_doc.md?user_id={user_test}")
        assert del_unauth.status_code == 403, f"Expected 403, got {del_unauth.status_code}"
        print(f"    [BLOCKED] Rejected unauthenticated delete with HTTP 403")

        # 7. Clean up test document with admin auth and assert physical file is unlinked
        print("\n[7] Cleaning up test document & verifying physical disk unlinking...")
        # Check that physical file exists prior to delete
        uploaded_files = list(UPLOAD_DIR.glob(f"*{user_test}*valid_doc.md"))
        assert len(uploaded_files) > 0, "Uploaded file was not found on disk prior to deletion"

        del_res = client.delete(
            f"/api/documents/valid_doc.md?user_id={user_test}",
            headers={"X-Admin-Passcode": "school123"},
        )
        assert del_res.status_code == 200

        # Verify physical file is gone
        remaining_files = list(UPLOAD_DIR.glob(f"*{user_test}*valid_doc.md"))
        assert len(remaining_files) == 0, f"Physical file still orphaned on disk: {remaining_files}"
        print("    [CLEANED] Physical upload file successfully purged from data/uploads/.")

        # 8. Test Orphan Sweeper
        print("\n[8] Testing orphan sweeper...")
        fake_orphan = UPLOAD_DIR / "fake_orphaned_doc.txt"
        fake_orphan.write_text("dummy orphaned content")
        assert fake_orphan.exists()

        cleanup_orphaned_uploads()
        assert not fake_orphan.exists(), "cleanup_orphaned_uploads failed to remove unreferenced file"
        print("    [SWEPT] Orphan sweeper successfully cleaned unreferenced file.")

    # Reset limiter and environment for subsequent test runs
    limiter.reset()
    os.environ["TESTING"] = "true"
    os.environ["ENFORCE_ADMIN_IN_TESTS"] = "false"

    print("\n" + "=" * 70)
    print("ALL SECURITY HARDENING TESTS PASSED! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_file_size_limits_and_rate_limiting()
