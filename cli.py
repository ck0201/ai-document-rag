"""
Command-Line Interface (CLI) for Document Q&A RAG Bot.

Usage:
  # Ingest a document
  python cli.py ingest data/company_handbook.md

  # Ask a single question
  python cli.py ask "What is the laptop and home office stipend?"

  # Launch interactive chat session
  python cli.py chat
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="AI Document Q&A Bot (Production RAG CLI)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and index a document (MD, TXT, PDF)")
    ingest_parser.add_argument("file", type=str, help="Path to document file")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question against indexed documents")
    ask_parser.add_argument("query", type=str, help="User question string")
    ask_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Launch interactive document Q&A chat loop")
    chat_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")

    args = parser.parse_args()
    pipeline = RAGPipeline()

    if args.command == "ingest":
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found at '{file_path}'")
            sys.exit(1)
        print(f"[*] Ingesting and indexing: {file_path}...")
        num_chunks = pipeline.ingest_document(file_path)
        print(f"[✓] Successfully indexed {num_chunks} chunks into Vector DB!")

    elif args.command == "ask":
        print(f"[*] Query: \"{args.query}\"")
        resp = pipeline.query(args.query, top_k=args.top_k)
        print("\n" + "=" * 50)
        print("BOT ANSWER:")
        print("=" * 50)
        print(resp.answer)
        print("\n" + "-" * 50)
        print("CITATIONS & RETRIEVED SOURCES:")
        for s in resp.sources:
            print(f"- {s['source']} (Page {s['page']}, Chunk #{s['chunk_index']}) | Similarity: {s['similarity']}")
        print("=" * 50)

    elif args.command == "chat":
        print("=" * 60)
        print("Interactive Document Q&A Bot (Type 'exit' or 'quit' to end)")
        print("=" * 60)
        while True:
            try:
                user_q = input("\nAsk a question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break

                resp = pipeline.query(user_q, top_k=args.top_k)
                print("\n[AI Answer]:")
                print(resp.answer)
                print("\n[Sources]:")
                for s in resp.sources:
                    print(f"  • {s['source']} (Page {s['page']}, Chunk #{s['chunk_index']}) - Score: {s['similarity']}")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break


if __name__ == "__main__":
    main()
