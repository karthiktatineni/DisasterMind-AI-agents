# Task 2P — Canonical Data Quality, Quarantine & Dedupe Report

- **Canonical `dataset_version` (SHA-256 prefix):** `3d35e5131d79a79d`
- **Raw Input Records:** 16,834
- **Canonical Snapshot Records:** 16,833
- **Quarantined Records:** 1

## Quarantine Counts by Reason Code

| Reason Code | Count |
|---|---|
| `TEMPORAL_INVERSION_END_BEFORE_START` | 1 |

## Deduplication Summary

- Exact `disaster_id` duplicates quarantined: 0
- Exact content-hash duplicates quarantined: 0
- Near-duplicate same-(type, country, date) clusters (retained as distinct EM-DAT events): 121

## Quality Flags Across Snapshot

| Quality Flag | Record Count |
|---|---|
| `event_name_synthesized` | 11,525 |
| `id_year_differs_from_start_year` | 53 |
| `location_placeholder_cleaned` | 20 |
| `missing_coordinates` | 14,979 |
| `missing_headline_impacts` | 401 |

## Canonical `search_text` Bounds (Snapshot)

- **Max characters:** 3,009 (record `2009-9323-IND`)
- **Max words:** 340
- **Max `cl100k_base` tokens:** 1,141 (record `2009-9323-IND`)
- **Min characters / tokens:** 111 chars / 37 tokens
- **Records containing literal `'unknown'`:** 0
- **Records with empty `search_text`:** 0

## Historical Stats Service Smoke Verification (from Canonical Parquet Snapshot)

- **Query:** `disaster_type='Storm', country='India'` -> status=`ok`, sample_size=`109`
- **Total Deaths (median / p90):** 23.0 / 94.8
- **Total Affected (median / p90):** 10000.0 / 1765000.0
