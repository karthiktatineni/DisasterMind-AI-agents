"""Preflight verification script for Pinecone vector index and NVIDIA NIM models.

Task 1P Gate:
1. Pinecone credentials valid; can describe/list indexes.
2. Index exists (creates serverless index if missing: metric=cosine, dimension matching embedding model).
3. Live NVIDIA embedding model responds (dimension matches index).
4. Live NVIDIA chat model responds.
5. Target namespace check: reports records or EMPTY_NAMESPACE (ready for Task 2P).
6. Refuses to start if dimension mismatch, wrong metric, or missing keys.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def fail(msg: str) -> None:
    print(f"\n[PREFLIGHT FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"[PREFLIGHT INFO] {msg}")


def ok(msg: str) -> None:
    print(f"[PREFLIGHT OK]   {msg}")


def run_preflight() -> int:
    print("=" * 70)
    print("DisasterMind AI — Task 1P Preflight Check")
    print("=" * 70)

    # 1. Check API Keys Presence
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "disastermind").strip()
    namespace = os.getenv("PINECONE_NAMESPACE", "v1-nemotron-3.5-embed-1b").strip()
    chat_model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip()
    embed_model = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b").strip()
    nvidia_base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")

    if not pinecone_key:
        fail("Missing PINECONE_API_KEY in .env")
    if not nvidia_key:
        fail("Missing NVIDIA_API_KEY in .env")

    ok("PINECONE_API_KEY and NVIDIA_API_KEY are configured.")

    # 2. Live NVIDIA Chat Model Check
    info(f"Testing live NVIDIA chat model: '{chat_model}'...")
    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        r_chat = requests.post(
            f"{nvidia_base_url}/chat/completions",
            headers=headers,
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": "respond with pong"}],
                "max_tokens": 50,
            },
            timeout=20,
        )
        if r_chat.status_code != 200:
            fail(f"Live NVIDIA chat model failed: HTTP {r_chat.status_code} - {r_chat.text[:120]}")
        chat_content = r_chat.json()["choices"][0]["message"].get("content") or ""
        ok(f"NVIDIA chat model responds live. Response snippet: '{chat_content.strip()[:30]}'")
    except Exception as exc:
        fail(f"NVIDIA chat model connection error: {exc}")

    # 3. Live NVIDIA Embedding Model Check
    info(f"Testing live NVIDIA embedding model: '{embed_model}'...")
    try:
        r_embed = requests.post(
            f"{nvidia_base_url}/embeddings",
            headers=headers,
            json={
                "model": embed_model,
                "input": ["Disaster emergency management preflight test"],
                "input_type": "query",
            },
            timeout=20,
        )
        if r_embed.status_code != 200:
            fail(f"Live NVIDIA embedding model failed: HTTP {r_embed.status_code} - {r_embed.text[:120]}")
        embedding_data = r_embed.json()["data"][0]["embedding"]
        embed_dim = len(embedding_data)
        ok(f"NVIDIA embedding model responds live. Vector dimension: {embed_dim}")
    except Exception as exc:
        fail(f"NVIDIA embedding model connection error: {exc}")

    # 4. Pinecone Connection & Index Verification / Creation
    info(f"Connecting to Pinecone and checking index: '{index_name}'...")
    try:
        pc = Pinecone(api_key=pinecone_key)
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        info(f"Existing indexes in account: {existing_indexes}")

        if index_name not in existing_indexes:
            info(f"Index '{index_name}' does not exist. Creating serverless index (cloud='aws', region='us-east-1', dim={embed_dim}, metric='cosine')...")
            pc.create_index(
                name=index_name,
                dimension=embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index ready
            for _ in range(60):
                desc = pc.describe_index(index_name)
                state = desc.status.get("state") if isinstance(desc.status, dict) else desc.status.state
                if state.lower() == "ready":
                    break
                time.sleep(2)
            ok(f"Index '{index_name}' created and ready.")
        else:
            desc = pc.describe_index(index_name)
            idx_dim = desc.dimension
            idx_metric = desc.metric
            if idx_metric.lower() != "cosine":
                fail(f"Index metric mismatch: expected 'cosine', found '{idx_metric}'.")
            if idx_dim != embed_dim:
                fail(
                    f"Index dimension mismatch: index '{index_name}' has dimension {idx_dim}, "
                    f"but embedding model '{embed_model}' produces dimension {embed_dim}."
                )
            ok(f"Index '{index_name}' exists with dimension={idx_dim} and metric='{idx_metric}'.")

    except Exception as exc:
        fail(f"Pinecone API error: {exc}")

    # 5. Target Namespace Check
    info(f"Inspecting index stats for namespace: '{namespace}'...")
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        total_vectors = stats.get("total_vector_count", 0)

        if namespace in namespaces and namespaces[namespace].get("vector_count", 0) > 0:
            count = namespaces[namespace]["vector_count"]
            ok(f"NAMESPACE_READY: target namespace '{namespace}' contains {count} vectors.")
        else:
            ok(f"EMPTY_NAMESPACE (ready for Task 2P): index has {total_vectors} total vectors across {len(namespaces)} namespaces.")

    except Exception as exc:
        fail(f"Failed to query index stats: {exc}")

    print("\n" + "=" * 70)
    print("TASK 1P PREFLIGHT: ALL GATES PASS")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_preflight())
