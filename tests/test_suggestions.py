"""
Test dynamic suggestions endpoint.
"""

from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api import app


def test_dynamic_suggestions():
    print("=" * 70)
    print("TESTING DYNAMIC PROMPT SUGGESTIONS ENDPOINT")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Fetch suggestions
        res = client.get("/api/suggestions")
        assert res.status_code == 200
        suggestions = res.json()
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 2, f"Expected at least 2 suggestions, got {len(suggestions)}"

        print(f"\n[+] Received {len(suggestions)} dynamic prompt suggestions:")
        for idx, s in enumerate(suggestions, 1):
            assert "icon" in s and "title" in s and "subtitle" in s and "query" in s
            assert len(s["query"]) > 5
            print(f"  {idx}. {s['icon']} {s['title']} -> '{s['query']}' ({s['subtitle']})")

    print("\n" + "=" * 70)
    print("SUGGESTIONS TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_dynamic_suggestions()
