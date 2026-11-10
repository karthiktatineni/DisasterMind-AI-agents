"""Start the real FastAPI app with opt-in httpx transport-level capture."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.wire_capture import enable_wire_capture

enable_wire_capture()

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("backend.app.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
