"""
Integration test for Server-Sent Events (SSE) streaming API endpoint.
"""

from pathlib import Path
import sys
import json
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import app


def test_sse_streaming():
    print("=" * 70)
    print("TESTING SERVER-SENT EVENTS (SSE) REAL-TIME STREAMING")
    print("=" * 70)

    with TestClient(app) as client:
        payload = {
            "query": "What are the laptop options and home office stipend?",
            "top_k": 2,
            "min_similarity": 0.35,
        }

        print(f"\n[+] POST /api/query/stream -> Question: '{payload['query']}'")
        with client.stream("POST", "/api/query/stream", json=payload) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            events_received = []
            tokens = []
            sources = []

            current_event = None
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    data_str = line.replace("data:", "").strip()
                    if current_event == "sources":
                        sources = json.loads(data_str)
                        print(f"  [SSE Event: 'sources'] Retrieved {len(sources)} source snippets:")
                        for s in sources:
                            print(f"     • {s['source']} (Chunk #{s['chunk_index']}, Score: {s['similarity']})")
                    elif current_event == "token":
                        token_obj = json.loads(data_str)
                        tokens.append(token_obj.get("token", ""))
                    elif current_event == "done":
                        print("  [SSE Event: 'done'] Stream finished successfully!")
                    events_received.append(current_event)

            full_streamed_text = "".join(tokens)
            print(f"\n[+] Total Tokens Streamed: {len(tokens)}")
            print(f"[+] Reconstructed Streamed Answer:\n{full_streamed_text[:250]}...\n")

            assert len(sources) > 0, "Expected sources in stream"
            assert len(tokens) > 0, "Expected tokens in stream"
            assert "done" in events_received, "Expected 'done' event in stream"

    print("=" * 70)
    print("STREAMING TESTS PASSED 100%! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_sse_streaming()
