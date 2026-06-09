#!/usr/bin/env python3
"""Compatibility entry point for uploading DisasterMind embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from database.upload_embeddings import upload  # noqa: E402


if __name__ == "__main__":
    upload()
