#!/usr/bin/env python3
"""Verify Supabase connection and generate connection report."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings  # noqa: E402


async def main() -> None:
    settings = get_settings()
    url = settings.supabase_url.rstrip("/")
    key = settings.supabase_service_role_key
    project_id = re.search(r"https://([^.]+)\.supabase\.co", url)
    project_id = project_id.group(1) if project_id else "unknown"

    lines = [
        "# Supabase Connection Report\n",
        f"- **Project ID:** {project_id}\n",
        f"- **URL configured:** {bool(url)}\n",
    ]

    if not url or not key:
        lines.append("- **Database status:** NOT CONFIGURED\n")
        lines.append("- **pgvector status:** UNKNOWN\n")
    else:
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                count_resp = await client.head(
                    f"{url}/rest/v1/disaster_embeddings?select=id",
                    headers={**headers, "Prefer": "count=exact"},
                )
                row_count = int(count_resp.headers.get("content-range", "0/0").split("/")[-1])
                lines.append("- **Database status:** REACHABLE\n")
                lines.append(f"- **Table row count:** {row_count}\n")
                lines.append("- **pgvector status:** ASSUMED ENABLED (verify via SQL)\n")
            except Exception as exc:
                lines.append(f"- **Database status:** ERROR ({exc})\n")

    (PROJECT_ROOT / "supabase_connection_report.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
