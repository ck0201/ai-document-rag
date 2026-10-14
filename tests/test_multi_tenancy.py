"""
Multi-Tenant Per-User Document Isolation Integration Tests.

Validates that:
1. User A and User B can upload documents simultaneously with full isolation.
2. User A cannot query or cite User B's documents.
3. User B cannot query or cite User A's documents.
4. User A only sees their own documents in the document list.
5. User B only sees their own documents in the document list.
6. Deleting a document for User A leaves User B's document completely intact.
"""

from pathlib import Path
import sys
import tempfile
import time

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import app


def test_multi_tenant_document_isolation():
    print("=" * 70)
    print("RUNNING MULTI-TENANT PER-USER DOCUMENT ISOLATION TESTS")
    print("=" * 70)

    user_a = "usr_alice_tenant_test"
    user_b = "usr_bob_tenant_test"

    with TestClient(app) as client:
        # Create distinct test documents for User A and User B
        with tempfile.TemporaryDirectory() as tmp_dir:
            path_a = Path(tmp_dir) / "alice_project_apollo.md"
            path_a.write_text(
                "# Project Apollo Specification\n\n"
                "Project Apollo has an approved total budget of 4.5 million dollars.\n"
                "The project lead is Commander Armstrong.\n"
                "Launch date is set for November 2026.\n",
                encoding="utf-8",
            )

            path_b = Path(tmp_dir) / "bob_project_zeus.md"
            path_b.write_text(
                "# Project Zeus Technical Architecture\n\n"
                "Project Zeus uses Rust and Kubernetes exclusively for microservices.\n"
                "The chief architect is Dr. Tesla.\n"
                "High-frequency trading latency target is 12 microseconds.\n",
                encoding="utf-8",
            )

            # 1. Upload User A's document with user_id=usr_alice
            print("\n[1] Uploading Document for User A (Alice)...")
            with path_a.open("rb") as f:
                res_a = client.post(
                    "/api/documents/upload",
                    data={"user_id": user_a},
                    files={"file": (path_a.name, f, "text/markdown")},
                )
            assert res_a.status_code == 202, f"Expected 202, got {res_a.status_code}"
            job_a_id = res_a.json()["job_id"]

            # 2. Upload User B's document with user_id=usr_bob
            print("\n[2] Uploading Document for User B (Bob)...")
            with path_b.open("rb") as f:
                res_b = client.post(
                    "/api/documents/upload",
                    data={"user_id": user_b},
                    files={"file": (path_b.name, f, "text/markdown")},
                )
            assert res_b.status_code == 202, f"Expected 202, got {res_b.status_code}"
            job_b_id = res_b.json()["job_id"]

            # 3. Wait for background ingestion jobs to complete
            print("\n[3] Waiting for both ingestion jobs to reach INDEXED...")
            for job_id in [job_a_id, job_b_id]:
                for _ in range(30):
                    job_res = client.get(f"/api/documents/jobs/{job_id}")
                    assert job_res.status_code == 200
                    if job_res.json()["status"] in ("INDEXED", "FAILED"):
                        break
                    time.sleep(0.3)
                assert job_res.json()["status"] == "INDEXED", f"Job {job_id} failed"

            # 4. Validate Isolated Document Lists
            print("\n[4] Validating Document List Isolation...")
            docs_a = client.get(f"/api/documents?user_id={user_a}").json()
            docs_b = client.get(f"/api/documents?user_id={user_b}").json()

            sources_a = [d["source"] for d in docs_a]
            sources_b = [d["source"] for d in docs_b]

            print(f"    User A's documents: {sources_a}")
            print(f"    User B's documents: {sources_b}")

            assert "alice_project_apollo.md" in sources_a
            assert "bob_project_zeus.md" not in sources_a, "User A must not see User B's documents"

            assert "bob_project_zeus.md" in sources_b
            assert "alice_project_apollo.md" not in sources_b, "User B must not see User A's documents"

            # 5. Query as User A on Project Apollo
            print("\n[5] User A queries Project Apollo budget...")
            query_res_a = client.post(
                "/api/query",
                json={
                    "query": "What is the approved total budget for Project Apollo?",
                    "user_id": user_a,
                    "top_k": 3,
                },
            )
            assert query_res_a.status_code == 200
            data_a = query_res_a.json()
            print(f"    User A Answer: {data_a['answer'][:90]}...")
            assert len(data_a["sources"]) > 0
            assert all(s["source"] == "alice_project_apollo.md" for s in data_a["sources"])

            # 6. User A tries querying User B's private project (Project Zeus)
            print("\n[6] User A attempts to query Project Zeus (User B's private data)...")
            leak_check = client.post(
                "/api/query",
                json={
                    "query": "What technology stack is used in Project Zeus?",
                    "user_id": user_a,
                    "top_k": 3,
                },
            )
            assert leak_check.status_code == 200
            leak_data = leak_check.json()
            # User A has no access to Project Zeus -> must not cite bob_project_zeus.md
            zeus_sources = [s for s in leak_data["sources"] if "bob_project_zeus" in s.get("source", "")]
            assert len(zeus_sources) == 0, "Data leak detected! User A retrieved User B's chunks"
            print("    [SECURE] User A retrieved 0 chunks from User B's documents.")

            # 7. User B queries Project Zeus
            print("\n[7] User B queries Project Zeus tech stack...")
            query_res_b = client.post(
                "/api/query",
                json={
                    "query": "What technology stack is used in Project Zeus?",
                    "user_id": user_b,
                    "top_k": 3,
                },
            )
            assert query_res_b.status_code == 200
            data_b = query_res_b.json()
            print(f"    User B Answer: {data_b['answer'][:90]}...")
            assert len(data_b["sources"]) > 0
            assert all(s["source"] == "bob_project_zeus.md" for s in data_b["sources"])

            # 8. User A deletes their document
            print("\n[8] User A deletes alice_project_apollo.md...")
            del_res_a = client.delete(f"/api/documents/alice_project_apollo.md?user_id={user_a}")
            assert del_res_a.status_code == 200

            # Verify User A's list is cleared
            docs_a_post = client.get(f"/api/documents?user_id={user_a}").json()
            assert not any(d["source"] == "alice_project_apollo.md" for d in docs_a_post if not d.get("is_sample"))

            # Verify User B's document is STILL present and untouched
            docs_b_post = client.get(f"/api/documents?user_id={user_b}").json()
            sources_b_post = [d["source"] for d in docs_b_post]
            assert "bob_project_zeus.md" in sources_b_post, "User B's document was corrupted by User A's deletion"
            print("    [ISOLATION VERIFIED] User B's document remains intact after User A deleted their own.")

            # 9. Cleanup User B's document
            client.delete(f"/api/documents/bob_project_zeus.md?user_id={user_b}")

    print("\n" + "=" * 70)
    print("ALL MULTI-TENANT ISOLATION TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_multi_tenant_document_isolation()
