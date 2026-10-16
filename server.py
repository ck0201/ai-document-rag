"""
Production Server Entry Point.

Can be run from ANY directory without PYTHONPATH or cd issues:
    python /home/frio/ai-document-rag/server.py
"""

import os
import sys
from pathlib import Path
import uvicorn

# Guarantee project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    is_prod = (
        bool(os.environ.get("RENDER"))
        or os.environ.get("ENV", "").lower() in ("production", "prod")
        or os.environ.get("PORT") is not None
    )
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=port,
        reload=not is_prod,
        app_dir=str(PROJECT_ROOT),
    )
