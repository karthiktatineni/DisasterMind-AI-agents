"""DisasterMind AI — Canonical Data Cleaning, Snapshot & Pinecone Ingestion Pipeline.

Task 2P:
1. Reconciles raw data against source records.
2. Validates schema and applies NFC normalization.
3. Separates quarantine records from cleaned records.
4. Generates versioned Parquet snapshot with computed historical stats.
5. Generates embeddings using configured NVIDIA model and upserts to Pinecone.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
