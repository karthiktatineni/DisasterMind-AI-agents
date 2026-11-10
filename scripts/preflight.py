"""Preflight verification script for Pinecone vector index, inference embeddings, and NVIDIA NIM chat.

Uses the exact same client libraries as the production backend:
- openai.AsyncOpenAI (httpx transport) for NVIDIA NIM chat (and NVIDIA NIM embeddings if selected)
- pinecone.Pinecone (httpx transport in pinecone>=10) for Pinecone index, stats & inference embeddings

Task 1P Gate:
1. Pinecone credentials valid; can describe/list indexes.
2. Index exists (creates serverless index if missing: metric=cosine, dimension matching embedding model).
3. Live configured embedding model responds (dimension matches index).
4. Live NVIDIA chat model responds.
5. Target namespace check: reports records or EMPTY_NAMESPACE (ready for Task 2P).
6. Refuses to start if dimension mismatch, wrong metric, blocked host, timeout, or missing/invalid keys.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def fail(msg: str) -> None:
    print(f"\n[PREFLIGHT FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"[PREFLIGHT INFO] {msg}")


def ok(msg: str) -> None:
    print(f"[PREFLIGHT OK]   {msg}")


PINECONE_INFERENCE_MODELS = frozenset({"llama-text-embed-v2", "multilingual-e5-large"})


async def _check_nvidia_async(
    *,
    api_key: str,
    base_url: str,
    chat_model: str,
    embed_model: str | None,
    timeout: float,
) -> int | None:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )
    try:
        info(f"Testing live NVIDIA chat model via AsyncOpenAI: '{chat_model}' (timeout={timeout}s)...")
        completion = await client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "respond with pong"}],
            max_tokens=50,
            stream=False,
        )
        chat_content = completion.choices[0].message.content or ""
        ok(f"NVIDIA chat model responds live via AsyncOpenAI. Snippet: '{chat_content.strip()[:30]}'")

        if embed_model is None:
            return None

        info(f"Testing live NVIDIA embedding model via AsyncOpenAI: '{embed_model}' (timeout={timeout}s)...")
        embed_resp = await client.embeddings.create(
            model=embed_model,
            input=["Disaster emergency management preflight test"],
            extra_body={
                "input_type": "query",
                "encoding_format": "float",
                "truncate": "END",
            },
        )
        embedding_data = embed_resp.data[0].embedding
        embed_dim = len(embedding_data)
        ok(f"NVIDIA embedding model responds live via AsyncOpenAI. Vector dimension: {embed_dim}")
        return embed_dim
    except APITimeoutError as exc:
        fail(f"NVIDIA request timed out (AsyncOpenAI timeout={timeout}s): {exc}")
    except APIConnectionError as exc:
        fail(f"NVIDIA connection error (AsyncOpenAI): {exc}")
    except APIError as exc:
        fail(f"NVIDIA API error (HTTP {getattr(exc, 'status_code', 'unknown')}): {exc}")
    except Exception as exc:
        fail(f"NVIDIA unexpected error: {exc}")
    finally:
        await client.close()
    return None


def run_preflight(control: str = "none") -> int:
    print("=" * 70)
    print(f"DisasterMind AI — Task 1P Preflight Check (control={control})")
    print("=" * 70)

    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "disastermind").strip()
    namespace = os.getenv("PINECONE_NAMESPACE", "v2-3d35e5131d79a79d-llama-text-embed-v2").strip()
    chat_model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip()
    embed_model = os.getenv("NVIDIA_EMBEDDING_MODEL", "llama-text-embed-v2").strip()
    nvidia_base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")

    client_timeout = 20.0
    pinecone_timeout = 20.0
    pinecone_host_override: str | None = None
    expected_dim_override: int | None = None

    if control == "bad-nvidia-key":
        nvidia_key = "nvapi-invalid-control-key"
    elif control == "bad-pinecone-key":
        pinecone_key = "pcsk_invalid_control_key"
    elif control == "wrong-dim":
        expected_dim_override = 4096
    elif control == "host-blocked":
        pinecone_host_override = "https://blocked-pinecone-host.invalid"
    elif control == "timeout":
        client_timeout = 0.001
        pinecone_timeout = 0.001
    elif control == "pinecone-timeout":
        pinecone_timeout = 0.001

    if not pinecone_key:
        fail("Missing PINECONE_API_KEY in environment / .env")
    if not nvidia_key:
        fail("Missing NVIDIA_API_KEY in environment / .env")

    ok("PINECONE_API_KEY and NVIDIA_API_KEY are configured.")

    # 1. Live NVIDIA Chat (+ Embedding if NVIDIA-hosted) checks via AsyncOpenAI (httpx)
    is_pinecone_embed = embed_model in PINECONE_INFERENCE_MODELS
    embed_dim = asyncio.run(
        _check_nvidia_async(
            api_key=nvidia_key,
            base_url=nvidia_base_url,
            chat_model=chat_model,
            embed_model=None if is_pinecone_embed else embed_model,
            timeout=client_timeout,
        )
    )

    # 2. Pinecone Connection (+ Embedding if Pinecone-hosted) & Index Verification via pinecone.Pinecone (httpx)
    info(f"Connecting to Pinecone via Pinecone SDK (timeout={pinecone_timeout}s) for index '{index_name}'...")
    try:
        pc_kwargs = {"api_key": pinecone_key, "timeout": pinecone_timeout}
        if pinecone_host_override:
            info(f"Control host-blocked routing Pinecone control/data plane to '{pinecone_host_override}'...")
            pc_kwargs["host"] = pinecone_host_override
        pc = Pinecone(**pc_kwargs)

        if is_pinecone_embed:
            info(f"Testing live Pinecone Inference embedding model: '{embed_model}' (timeout={pinecone_timeout}s)...")
            inf_pc = Pinecone(api_key=pinecone_key, timeout=pinecone_timeout)
            inf_res = inf_pc.inference.embed(
                model=embed_model,
                inputs=["Disaster emergency management preflight test"],
                parameters={"input_type": "query", "truncate": "END"},
            )
            embed_dim = len(inf_res.data[0].values)
            ok(f"Pinecone Inference embedding model responds live. Vector dimension: {embed_dim}")

        assert embed_dim is not None
        if expected_dim_override is not None:
            info(f"Control wrong-dim overriding reported embedding dimension to {expected_dim_override}.")
            embed_dim = expected_dim_override

        existing_indexes = [idx.name for idx in pc.list_indexes()]
        info(f"Existing indexes in account: {existing_indexes}")

        if index_name not in existing_indexes:
            info(
                f"Index '{index_name}' does not exist. Creating serverless index "
                f"(cloud='aws', region='us-east-1', dim={embed_dim}, metric='cosine')..."
            )
            pc.create_index(
                name=index_name,
                dimension=embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
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
                    f"but embedding model '{embed_model}' requires dimension {embed_dim}."
                )
            ok(f"Index '{index_name}' exists with dimension={idx_dim} and metric='{idx_metric}'.")
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"Pinecone control/connection error: {type(exc).__name__}: {exc}")

    # 3. Target Namespace Data-Plane Check
    info(f"Inspecting index stats for namespace '{namespace}'...")
    try:
        if pinecone_host_override:
            index = pc.Index(host=pinecone_host_override)
        else:
            index = pc.Index(index_name)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        total_vectors = stats.get("total_vector_count", 0)

        if namespace in namespaces and namespaces[namespace].get("vector_count", 0) > 0:
            count = namespaces[namespace]["vector_count"]
            ok(f"NAMESPACE_READY: target namespace '{namespace}' contains {count} vectors.")
        else:
            ok(
                f"EMPTY_NAMESPACE (ready for Task 2P): index has {total_vectors} total vectors "
                f"across {len(namespaces)} namespaces."
            )
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"Failed to query index stats: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("TASK 1P PREFLIGHT: ALL GATES PASS")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 1P Preflight Check & Negative Controls")
    parser.add_argument(
        "--control",
        choices=["none", "bad-nvidia-key", "bad-pinecone-key", "wrong-dim", "host-blocked", "timeout", "pinecone-timeout"],
        default="none",
        help="Run a negative control (default: none = normal live check)",
    )
    args = parser.parse_args()
    return run_preflight(control=args.control)


if __name__ == "__main__":
    sys.exit(main())
