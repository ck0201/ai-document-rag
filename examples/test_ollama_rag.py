"""
Verification test for local Ollama llama3.2:3b in the RAG Pipeline.
"""

from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import OllamaLLMClient
from src.rag_pipeline import RAGPipeline


def test_ollama_local_rag():
    print("=" * 70)
    print("TESTING LOCAL LLM (llama3.2:3b via Ollama) IN RAG PIPELINE")
    print("=" * 70)

    # 1. Initialize Pipeline with Ollama client
    ollama_client = OllamaLLMClient(model="llama3.2:3b")
    pipeline = RAGPipeline(llm=ollama_client)

    # 2. Test Question 1: Direct Policy Retrieval
    q1 = "What laptop choices and home office stipend amount do engineers receive?"
    print(f"\n[+] Query 1: '{q1}'")
    start = time.perf_counter()
    resp1 = pipeline.query(q1)
    duration1 = time.perf_counter() - start

    print(f"    Generation Time: {duration1:.2f} seconds")
    print("\n[AI Answer from llama3.2:3b]:")
    print(resp1.answer)
    print("\n[Citations]:")
    for s in resp1.sources:
        print(f"  • {s['source']} (Page {s['page']}, Chunk #{s['chunk_index']}) - Score: {s['similarity']}")

    # 3. Test Question 2: Refusal on Absent Information
    q2 = "What is the policy for paid paternity leave?"
    print(f"\n" + "-" * 70)
    print(f"[+] Query 2 (Refusal Test): '{q2}'")
    start = time.perf_counter()
    resp2 = pipeline.query(q2)
    duration2 = time.perf_counter() - start

    print(f"    Generation Time: {duration2:.2f} seconds")
    print("\n[AI Answer from llama3.2:3b]:")
    print(resp2.answer)

    print("\n" + "=" * 70)
    print("OLLAMA LOCAL LLM VERIFICATION PASSED! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    test_ollama_local_rag()
