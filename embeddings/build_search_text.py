"""Build search_text for disaster records."""

from __future__ import annotations

import math
from typing import Any


def _safe_str(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "unknown"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "unknown"


def _safe_number(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "unknown"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)
    return _safe_str(value)


def build_search_text(record: dict[str, Any]) -> str:
    event = _safe_str(record.get("event_name"))
    if event == "unknown":
        event = _safe_str(record.get("disaster_id"))

    return (
        f"Event: {event}\n"
        f"Disaster Type: {_safe_str(record.get('disaster_type'))}\n"
        f"Disaster Subtype: {_safe_str(record.get('disaster_subtype'))}\n"
        f"Country: {_safe_str(record.get('country'))}\n"
        f"Region: {_safe_str(record.get('region'))}\n"
        f"Location: {_safe_str(record.get('location'))}\n"
        f"Year: {_safe_number(record.get('start_year'))}\n"
        f"Deaths: {_safe_number(record.get('total_deaths'))}\n"
        f"Affected Population: {_safe_number(record.get('total_affected'))}\n"
        f"Damage USD Thousands: {_safe_number(record.get('total_damage_usd_thousands'))}\n"
        f"Magnitude: {_safe_number(record.get('magnitude'))} "
        f"{_safe_str(record.get('magnitude_scale'))}"
    )
