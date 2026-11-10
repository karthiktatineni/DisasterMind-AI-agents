"""Helpers for respecting upstream Retry-After response headers."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from collections.abc import Mapping


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Return a valid Retry-After (or Retry-After-Ms) duration, without a cap.

    Returning the full duration is intentional: callers must not retry before the
    provider's requested time. Invalid or absent headers return None.
    """
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    retry_ms = normalized.get("retry-after-ms")
    if retry_ms is not None:
        try:
            return max(0.0, float(retry_ms) / 1000.0)
        except (TypeError, ValueError):
            pass

    value = normalized.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
