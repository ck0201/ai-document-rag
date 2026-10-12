"""
Production Server Entry Point.

Can be run from ANY directory without PYTHONPATH or cd issues:
    python /home/frio/ai-document-rag/server.py
"""

import sys
from pathlib import Path
import uvicorn

# Guarantee project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(PROJECT_ROOT),
    )
