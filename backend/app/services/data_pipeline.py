"""Canonical data reconciliation, cleaning, quarantine, deduplication, and Parquet snapshot builder."""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from backend.app.services.embedding_service import count_tokens_approx

FILLER_LITERALS = {"unknown", "null", "n/a", "na", "none", "nan", "-", "--", ""}

FIELD_MAP_JSON_TO_XLSX = {
    "disaster_id": "DisNo.",
    "disaster_group": "Disaster Group",
    "disaster_subgroup": "Disaster Subgroup",
    "disaster_type": "Disaster Type",
    "disaster_subtype": "Disaster Subtype",
    "event_name": "Event Name",
    "country": "Country",
    "region": "Region",
    "location": "Location",
    "magnitude": "Magnitude",
    "magnitude_scale": "Magnitude Scale",
    "start_year": "Start Year",
    "start_month": "Start Month",
    "start_day": "Start Day",
    "end_year": "End Year",
    "end_month": "End Month",
    "end_day": "End Day",
    "total_deaths": "Total Deaths",
    "injured": "No. Injured",
    "affected": "No. Affected",
    "homeless": "No. Homeless",
    "total_affected": "Total Affected",
    "total_damage_usd_thousands": "Total Damage ('000 US$)",
    "latitude": "Latitude",
    "longitude": "Longitude",
}

UNIT_CANONICAL = {
    "km2": "Km2",
    "kph": "Kph",
    "m3": "m3",
    "moment magnitude": "Moment Magnitude",
    "°c": "°C",
    "c": "°C",
    "vaccinated": "Vaccinated",
}

CATEGORY_TAG_MAP = {
    "Storm": "meteorological",
    "Extreme temperature": "meteorological",
    "Flood": "hydrological",
    "Glacial lake outburst flood": "hydrological",
    "Mass movement (wet)": "hydrological",
    "Earthquake": "geophysical",
    "Volcanic activity": "geophysical",
    "Mass movement (dry)": "geophysical",
    "Drought": "climatological",
    "Wildfire": "climatological",
    "Epidemic": "biological",
    "Infestation": "biological",
    "Animal incident": "biological",
    "Impact": "extra-terrestrial",
}

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NAME_UNKNOWN_PARENS_RE = re.compile(
    r"\(\s*Name\s+Unknown(?:\s*\(\d+\))?(?:\s+district)?(?:\s*\(\d+\))?(?:\s*,\s*)?",
    re.IGNORECASE,
)
_NAME_UNKNOWN_RE = re.compile(
    r"\bName\s+Unknown(?:\s*\(\d+\))?(?:\s+district)?(?:\s*\(\d+\))?",
    re.IGNORECASE,
)


def _col_to_idx(ref: str) -> int:
    col = "".join(c for c in ref if c.isalpha())
    idx = 0
    for c in col:
        idx = idx * 26 + (ord(c.upper()) - ord("A") + 1)
    return idx - 1


def read_emdat_xlsx_rows(xlsx_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read the primary 'EM-DAT Data' sheet from an EM-DAT Excel export using stdlib XML."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.findall("m:si", ns):
                shared.append(
                    "".join(
                        t.text or ""
                        for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    )
                )
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[Any]] = []
        for row_el in root.findall(".//m:sheetData/m:row", ns):
            row_dict: dict[int, Any] = {}
            max_c = 0
            for c_el in row_el.findall("m:c", ns):
                ci = _col_to_idx(c_el.attrib.get("r", "A1"))
                max_c = max(max_c, ci)
                cell_type = c_el.attrib.get("t", "")
                val: Any = None
                if cell_type == "inlineStr":
                    is_el = c_el.find("m:is", ns)
                    if is_el is not None:
                        val = "".join(
                            x.text or ""
                            for x in is_el.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                        )
                else:
                    v_el = c_el.find("m:v", ns)
                    if v_el is not None and v_el.text is not None:
                        if cell_type == "s":
                            val = shared[int(v_el.text)]
                        elif cell_type == "b":
                            val = v_el.text == "1"
                        elif cell_type == "str":
                            val = v_el.text
                        else:
                            try:
                                fval = float(v_el.text)
                                val = int(fval) if fval.is_integer() and "." not in v_el.text else fval
                            except ValueError:
                                val = v_el.text
                row_dict[ci] = val
            rows.append([row_dict.get(i) for i in range(max_c + 1)])

    headers = [str(h) for h in rows[0]]
    records: list[dict[str, Any]] = []
    for row in rows[1:]:
        records.append({headers[i]: (row[i] if i < len(row) else None) for i in range(len(headers))})
    return headers, records


def _norm_compare_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 6)
    if isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = unicodedata.normalize("NFC", v).strip()
        return s if s else None
    return v


def reconcile_curated_vs_xlsx(curated_path: Path, xlsx_path: Path) -> dict[str, Any]:
    """Step 1: Reconcile curated JSON dataset against the EM-DAT Excel source."""
    headers, xlsx_rows = read_emdat_xlsx_rows(xlsx_path)
    curated_rows = json.loads(curated_path.read_text(encoding="utf-8"))

    xlsx_by_id: dict[str, dict[str, Any]] = {}
    xlsx_dups: list[str] = []
    for row in xlsx_rows:
        did = _norm_compare_value(row.get("DisNo."))
        if did is not None:
            if did in xlsx_by_id:
                xlsx_dups.append(str(did))
            xlsx_by_id[str(did)] = row

    curated_by_id: dict[str, dict[str, Any]] = {}
    curated_dups: list[str] = []
    for row in curated_rows:
        did = _norm_compare_value(row.get("disaster_id"))
        if did is not None:
            if did in curated_by_id:
                curated_dups.append(str(did))
            curated_by_id[str(did)] = row

    only_in_xlsx = sorted(set(xlsx_by_id) - set(curated_by_id))
    only_in_curated = sorted(set(curated_by_id) - set(xlsx_by_id))
    shared_ids = sorted(set(xlsx_by_id) & set(curated_by_id))

    disagreement_counts: Counter[str] = Counter()
    disagreement_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for did in shared_ids:
        xr = xlsx_by_id[did]
        cr = curated_by_id[did]
        for json_key, xlsx_col in FIELD_MAP_JSON_TO_XLSX.items():
            if xlsx_col not in headers:
                continue
            cv = _norm_compare_value(cr.get(json_key))
            xv = _norm_compare_value(xr.get(xlsx_col))
            if cv != xv:
                disagreement_counts[json_key] += 1
                if len(disagreement_examples[json_key]) < 3:
                    disagreement_examples[json_key].append(
                        {"disaster_id": did, "curated": cv, "xlsx": xv}
                    )

    return {
        "curated_count": len(curated_rows),
        "xlsx_count": len(xlsx_rows),
        "curated_unique_ids": len(curated_by_id),
        "xlsx_unique_ids": len(xlsx_by_id),
        "curated_duplicate_ids": curated_dups,
        "xlsx_duplicate_ids": xlsx_dups,
        "shared_id_count": len(shared_ids),
        "only_in_xlsx_count": len(only_in_xlsx),
        "only_in_xlsx_sample": only_in_xlsx[:10],
        "only_in_curated_count": len(only_in_curated),
        "only_in_curated_sample": only_in_curated[:10],
        "mapped_fields_checked": len(FIELD_MAP_JSON_TO_XLSX),
        "field_disagreements": dict(disagreement_counts),
        "disagreement_examples": dict(disagreement_examples),
        "total_field_disagreements": sum(disagreement_counts.values()),
    }


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    text = unicodedata.normalize("NFC", str(value))
    text = _CTRL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in FILLER_LITERALS:
        return None
    return text


def _clean_location(value: Any) -> tuple[str | None, bool]:
    text = _clean_text(value)
    if text is None:
        return None, False
    had_placeholder = False
    if "unknown" in text.lower():
        had_placeholder = True
        text = _NAME_UNKNOWN_PARENS_RE.sub("(", text)
        text = _NAME_UNKNOWN_RE.sub("", text)
        text = re.sub(r"\(\s*,?\s*", "(", text)
        text = re.sub(r",\s*\)", ")", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\s*,\s*,+", ", ", text)
        text = re.sub(r"\s+", " ", text).strip(" ,;")
        text = text if text and text.lower() not in FILLER_LITERALS else None
    if text and text.startswith(","):
        text = text.lstrip(" ,;") or None
    return text, had_placeholder


def _clean_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(fval) or math.isinf(fval):
        return None
    return fval


def _clean_int(value: Any) -> int | None:
    fval = _clean_float(value)
    if fval is None:
        return None
    if not fval.is_integer():
        return None
    return int(fval)


def _normalize_unit(scale: str | None) -> str | None:
    if scale is None:
        return None
    return UNIT_CANONICAL.get(scale.lower(), scale)


def _derive_category_tag(disaster_type: str, disaster_subgroup: str | None) -> str:
    if disaster_type in CATEGORY_TAG_MAP:
        return CATEGORY_TAG_MAP[disaster_type]
    if disaster_subgroup and disaster_subgroup.lower() in {
        "meteorological",
        "hydrological",
        "geophysical",
        "climatological",
        "biological",
        "extra-terrestrial",
    }:
        return disaster_subgroup.lower()
    return "technological"


def build_clean_search_text(record: dict[str, Any]) -> str:
    """Build deterministic clean search_text omitting nulls and containing zero 'unknown' literals."""
    lines = [
        f"Event: {record['event_name']}",
        f"Disaster Type: {record['disaster_type']}",
    ]
    if record.get("disaster_subtype"):
        lines.append(f"Disaster Subtype: {record['disaster_subtype']}")
    lines.append(f"Country: {record['country']}")
    if record.get("region"):
        lines.append(f"Region: {record['region']}")
    if record.get("location"):
        lines.append(f"Location: {record['location']}")

    sy = record["start_year"]
    sm = record.get("start_month")
    sd = record.get("start_day")
    date_parts = [str(sy)]
    if sm is not None:
        date_parts.append(f"{sm:02d}")
        if sd is not None:
            date_parts.append(f"{sd:02d}")
    lines.append(f"Date: {'-'.join(date_parts)}")

    if record.get("total_deaths") is not None:
        lines.append(f"Deaths: {int(record['total_deaths']) if float(record['total_deaths']).is_integer() else record['total_deaths']}")
    if record.get("total_affected") is not None:
        lines.append(f"Affected Population: {int(record['total_affected']) if float(record['total_affected']).is_integer() else record['total_affected']}")
    if record.get("total_damage_usd_thousands") is not None:
        dmg = record["total_damage_usd_thousands"]
        lines.append(f"Damage USD Thousands: {int(dmg) if float(dmg).is_integer() else dmg}")
    if record.get("magnitude") is not None:
        mag = record["magnitude"]
        mag_str = str(int(mag) if float(mag).is_integer() else mag)
        if record.get("magnitude_scale"):
            mag_str = f"{mag_str} {record['magnitude_scale']}"
        lines.append(f"Magnitude: {mag_str}")
    return "\n".join(lines)


def validate_and_clean_record(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
    """Steps 2-4: Normalize, validate schema/dates/physics, or return quarantine (reason_code, detail)."""
    disaster_id = _clean_text(raw.get("disaster_id"))
    disaster_type = _clean_text(raw.get("disaster_type"))
    country = _clean_text(raw.get("country"))
    start_year = _clean_int(raw.get("start_year"))

    if not disaster_id or not disaster_type or not country or start_year is None:
        return None, (
            "MISSING_REQUIRED_FIELD",
            f"disaster_id={disaster_id!r}, disaster_type={disaster_type!r}, country={country!r}, start_year={raw.get('start_year')!r}",
        )

    if not (1900 <= start_year <= 2026):
        return None, ("INVALID_YEAR_RANGE", f"start_year={start_year} outside [1900, 2026]")

    start_month = _clean_int(raw.get("start_month"))
    start_day = _clean_int(raw.get("start_day"))
    end_year = _clean_int(raw.get("end_year"))
    end_month = _clean_int(raw.get("end_month"))
    end_day = _clean_int(raw.get("end_day"))

    for label, yr, mo, dy in (
        ("start", start_year, start_month, start_day),
        ("end", end_year, end_month, end_day),
    ):
        if mo is not None and not (1 <= mo <= 12):
            return None, ("INVALID_CALENDAR_DATE", f"{label}_month={mo}")
        if dy is not None:
            if mo is not None and yr is not None:
                max_day = calendar.monthrange(yr, mo)[1]
                if not (1 <= dy <= max_day):
                    return None, ("INVALID_CALENDAR_DATE", f"{label} {yr}-{mo}-{dy} exceeds month days {max_day}")
            elif not (1 <= dy <= 31):
                return None, ("INVALID_CALENDAR_DATE", f"{label}_day={dy}")

    if end_year is not None:
        s_tuple = (start_year, start_month or 1, start_day or 1)
        e_tuple = (end_year, end_month or 12, end_day or 31)
        if e_tuple < s_tuple:
            return None, (
                "TEMPORAL_INVERSION_END_BEFORE_START",
                f"start={s_tuple} > end={e_tuple} for disaster_id={disaster_id}",
            )

    lat = _clean_float(raw.get("latitude"))
    lon = _clean_float(raw.get("longitude"))
    if (lat is not None and lon is None) or (lon is not None and lat is None):
        return None, ("PARTIAL_COORDINATES", f"lat={lat}, lon={lon}")
    if lat is not None and not (-90.0 <= lat <= 90.0):
        return None, ("INVALID_COORDINATES", f"latitude={lat} out of [-90, 90]")
    if lon is not None and not (-180.0 <= lon <= 180.0):
        return None, ("INVALID_COORDINATES", f"longitude={lon} out of [-180, 180]")

    numeric_fields = (
        "total_deaths",
        "injured",
        "affected",
        "homeless",
        "total_affected",
        "total_damage_usd_thousands",
    )
    cleaned_nums: dict[str, float | None] = {}
    for nf in numeric_fields:
        val = _clean_float(raw.get(nf))
        if val is not None and val < 0:
            return None, ("NEGATIVE_IMPACT_VALUE", f"{nf}={val}")
        cleaned_nums[nf] = val

    magnitude = _clean_float(raw.get("magnitude"))
    magnitude_scale = _normalize_unit(_clean_text(raw.get("magnitude_scale")))
    if magnitude is None:
        magnitude_scale = None
    elif magnitude_scale == "Moment Magnitude" and not (1.0 <= magnitude <= 10.0):
        return None, ("IMPLAUSIBLE_MAGNITUDE", f"Moment Magnitude={magnitude}")
    elif magnitude_scale == "Kph" and not (0.0 < magnitude <= 450.0):
        return None, ("IMPLAUSIBLE_MAGNITUDE", f"Kph={magnitude}")
    elif magnitude_scale == "°C" and not (-85.0 <= magnitude <= 65.0):
        return None, ("IMPLAUSIBLE_MAGNITUDE", f"°C={magnitude}")
    elif magnitude_scale in {"Km2", "m3", "Vaccinated"} and magnitude <= 0.0:
        return None, ("IMPLAUSIBLE_MAGNITUDE", f"{magnitude_scale}={magnitude}")

    raw_event_name = _clean_text(raw.get("event_name"))
    synthesized_name = False
    if raw_event_name is None:
        event_name = f"{disaster_type} - {country} ({start_year})"
        synthesized_name = True
    else:
        event_name = raw_event_name

    location, had_loc_placeholder = _clean_location(raw.get("location"))
    disaster_group = _clean_text(raw.get("disaster_group"))
    disaster_subgroup = _clean_text(raw.get("disaster_subgroup"))
    disaster_subtype = _clean_text(raw.get("disaster_subtype"))
    region = _clean_text(raw.get("region"))

    quality_flags: list[str] = []
    if synthesized_name:
        quality_flags.append("event_name_synthesized")
    if had_loc_placeholder:
        quality_flags.append("location_placeholder_cleaned")
    try:
        id_year = int(disaster_id.split("-")[0])
        if id_year != start_year:
            quality_flags.append("id_year_differs_from_start_year")
    except (ValueError, IndexError):
        quality_flags.append("nonstandard_id_format")
    if all(cleaned_nums[k] is None for k in ("total_deaths", "total_affected", "total_damage_usd_thousands")):
        quality_flags.append("missing_headline_impacts")
    if lat is None or lon is None:
        quality_flags.append("missing_coordinates")

    display_label = (
        f"{event_name} ({start_year})"
        if str(start_year) not in event_name
        else event_name
    )
    if country.lower() not in display_label.lower():
        display_label = f"{display_label} — {country}"

    cleaned: dict[str, Any] = {
        "disaster_id": disaster_id,
        "disaster_group": disaster_group,
        "disaster_subgroup": disaster_subgroup,
        "disaster_type": disaster_type,
        "disaster_subtype": disaster_subtype,
        "category_tag": _derive_category_tag(disaster_type, disaster_subgroup),
        "event_name": event_name,
        "event_name_synthesized": synthesized_name,
        "display_label": display_label,
        "country": country,
        "region": region,
        "location": location,
        "magnitude": magnitude,
        "magnitude_scale": magnitude_scale,
        "start_year": start_year,
        "start_month": start_month,
        "start_day": start_day,
        "end_year": end_year,
        "end_month": end_month,
        "end_day": end_day,
        "total_deaths": cleaned_nums["total_deaths"],
        "injured": cleaned_nums["injured"],
        "affected": cleaned_nums["affected"],
        "homeless": cleaned_nums["homeless"],
        "total_affected": cleaned_nums["total_affected"],
        "total_damage_usd_thousands": cleaned_nums["total_damage_usd_thousands"],
        "latitude": lat,
        "longitude": lon,
        "quality_flags": quality_flags,
    }
    search_text = build_clean_search_text(cleaned)
    if not search_text.strip():
        return None, ("EMPTY_SEARCH_TEXT", "Constructed search_text is empty")
    if "unknown" in search_text.lower():
        return None, ("FILLER_LITERAL_REMAINS", f"search_text still contains 'unknown': {search_text[:120]}")

    cleaned["search_text"] = search_text
    cleaned["search_text_sha256"] = hashlib.sha256(search_text.encode("utf-8")).hexdigest()
    cleaned["search_text_chars"] = len(search_text)
    cleaned["search_text_tokens_approx"] = count_tokens_approx(search_text)
    return cleaned, None


def compute_content_hash(record: dict[str, Any]) -> str:
    """Deterministic SHA-256 over core event payload (ignoring disaster_id) for exact dedupe."""
    payload = {
        k: record[k]
        for k in (
            "disaster_type",
            "disaster_subtype",
            "country",
            "location",
            "start_year",
            "start_month",
            "start_day",
            "total_deaths",
            "total_affected",
            "total_damage_usd_thousands",
            "search_text",
        )
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def run_cleaning_and_snapshot(
    curated_path: Path,
    xlsx_path: Path,
    snapshot_path: Path,
    quarantine_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run Steps 1-5 end-to-end and return summary telemetry."""
    reconciliation = reconcile_curated_vs_xlsx(curated_path, xlsx_path)
    raw_records = json.loads(curated_path.read_text(encoding="utf-8"))

    cleaned_records: list[dict[str, Any]] = []
    quarantine_records: list[dict[str, Any]] = []
    quarantine_counts: Counter[str] = Counter()

    seen_ids: set[str] = set()
    seen_content_hashes: dict[str, str] = {}
    near_dup_groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)

    for raw in raw_records:
        cleaned, err = validate_and_clean_record(raw)
        if err is not None or cleaned is None:
            code, detail = err  # type: ignore[misc]
            quarantine_counts[code] += 1
            quarantine_records.append(
                {
                    "disaster_id": raw.get("disaster_id"),
                    "reason_code": code,
                    "detail": detail,
                    "raw_record": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in raw.items()},
                }
            )
            continue

        did = cleaned["disaster_id"]
        if did in seen_ids:
            quarantine_counts["DUPLICATE_DISASTER_ID"] += 1
            quarantine_records.append(
                {
                    "disaster_id": did,
                    "reason_code": "DUPLICATE_DISASTER_ID",
                    "detail": f"Duplicate disaster_id {did}",
                    "raw_record": cleaned,
                }
            )
            continue
        seen_ids.add(did)

        chash = compute_content_hash(cleaned)
        if chash in seen_content_hashes:
            quarantine_counts["DUPLICATE_CONTENT_HASH"] += 1
            quarantine_records.append(
                {
                    "disaster_id": did,
                    "reason_code": "DUPLICATE_CONTENT_HASH",
                    "detail": f"Exact content duplicate of {seen_content_hashes[chash]}",
                    "raw_record": cleaned,
                }
            )
            continue
        seen_content_hashes[chash] = did

        if cleaned["start_month"] is not None and cleaned["start_day"] is not None:
            near_key = (
                cleaned["disaster_type"],
                cleaned["country"],
                cleaned["start_year"],
                cleaned["start_month"],
                cleaned["start_day"],
            )
            near_dup_groups[near_key].append(did)

        cleaned_records.append(cleaned)

    cleaned_records.sort(key=lambda r: r["disaster_id"])

    # Compute deterministic dataset_version hash over canonical records
    version_hasher = hashlib.sha256()
    for rec in cleaned_records:
        version_hasher.update(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        version_hasher.update(b"\n")
    dataset_version = version_hasher.hexdigest()[:16]

    for rec in cleaned_records:
        rec["dataset_version"] = dataset_version

    # Write quarantine JSON
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text(
        json.dumps(
            {
                "total_quarantined": len(quarantine_records),
                "counts_by_reason": dict(quarantine_counts),
                "records": quarantine_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Write canonical Parquet snapshot with embedded schema metadata
    arrow_schema = pa.schema(
        [
            pa.field("disaster_id", pa.string()),
            pa.field("dataset_version", pa.string()),
            pa.field("disaster_group", pa.string()),
            pa.field("disaster_subgroup", pa.string()),
            pa.field("disaster_type", pa.string()),
            pa.field("disaster_subtype", pa.string()),
            pa.field("category_tag", pa.string()),
            pa.field("event_name", pa.string()),
            pa.field("event_name_synthesized", pa.bool_()),
            pa.field("display_label", pa.string()),
            pa.field("country", pa.string()),
            pa.field("region", pa.string()),
            pa.field("location", pa.string()),
            pa.field("magnitude", pa.float64()),
            pa.field("magnitude_scale", pa.string()),
            pa.field("start_year", pa.int32()),
            pa.field("start_month", pa.int32()),
            pa.field("start_day", pa.int32()),
            pa.field("end_year", pa.int32()),
            pa.field("end_month", pa.int32()),
            pa.field("end_day", pa.int32()),
            pa.field("total_deaths", pa.float64()),
            pa.field("injured", pa.float64()),
            pa.field("affected", pa.float64()),
            pa.field("homeless", pa.float64()),
            pa.field("total_affected", pa.float64()),
            pa.field("total_damage_usd_thousands", pa.float64()),
            pa.field("latitude", pa.float64()),
            pa.field("longitude", pa.float64()),
            pa.field("quality_flags", pa.list_(pa.string())),
            pa.field("search_text", pa.string()),
            pa.field("search_text_sha256", pa.string()),
            pa.field("search_text_chars", pa.int32()),
            pa.field("search_text_tokens_approx", pa.int32()),
        ],
        metadata={
            b"dataset_version": dataset_version.encode("utf-8"),
            b"record_count": str(len(cleaned_records)).encode("utf-8"),
            b"source": b"EM-DAT, CRED / UCLouvain, Brussels, Belgium (2026-05-20)",
        },
    )
    table = pa.Table.from_pylist(cleaned_records, schema=arrow_schema)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, snapshot_path, compression="snappy")

    max_chars_rec = max(cleaned_records, key=lambda r: r["search_text_chars"])
    max_toks_rec = max(cleaned_records, key=lambda r: r["search_text_tokens_approx"])
    flag_counts: Counter[str] = Counter()
    for rec in cleaned_records:
        for fl in rec["quality_flags"]:
            flag_counts[fl] += 1

    near_dup_clusters = [
        {"key": list(k), "ids": v}
        for k, v in near_dup_groups.items()
        if len(v) > 1
    ]

    summary = {
        "dataset_version": dataset_version,
        "raw_count": len(raw_records),
        "cleaned_count": len(cleaned_records),
        "quarantined_count": len(quarantine_records),
        "quarantine_counts_by_reason": dict(quarantine_counts),
        "reconciliation": reconciliation,
        "dedupe": {
            "exact_id_duplicates": quarantine_counts.get("DUPLICATE_DISASTER_ID", 0),
            "exact_content_duplicates": quarantine_counts.get("DUPLICATE_CONTENT_HASH", 0),
            "near_duplicate_same_day_country_type_clusters": len(near_dup_clusters),
            "near_duplicate_sample": near_dup_clusters[:10],
        },
        "quality_flag_counts": dict(flag_counts),
        "search_text_stats": {
            "max_chars": max_chars_rec["search_text_chars"],
            "max_chars_id": max_chars_rec["disaster_id"],
            "max_words": max(len(r["search_text"].split()) for r in cleaned_records),
            "max_cl100k_tokens": max_toks_rec["search_text_tokens_approx"],
            "max_cl100k_tokens_id": max_toks_rec["disaster_id"],
            "min_chars": min(r["search_text_chars"] for r in cleaned_records),
            "min_cl100k_tokens": min(r["search_text_tokens_approx"] for r in cleaned_records),
            "contains_unknown_literal": sum(1 for r in cleaned_records if "unknown" in r["search_text"].lower()),
            "empty_search_text": sum(1 for r in cleaned_records if not r["search_text"].strip()),
        },
        "snapshot_path": str(snapshot_path),
        "quarantine_path": str(quarantine_path),
    }
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

