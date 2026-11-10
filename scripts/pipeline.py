"""DisasterMind AI — Canonical Data Cleaning, Snapshot, Model Selection & Pinecone Ingestion Pipeline.

Task 2P stages (strict order):
1. clean: Reconcile curated dataset vs EM-DAT xlsx, validate/normalize, quarantine, dedupe,
          synthesize labels/search_text, build canonical Parquet snapshot with dataset_version,
          and produce reconciliation + data quality reports.
2. select-model: Run pre-registered DEV benchmark across candidate embedding models (cached locally).
3. ingest: Embed canonical snapshot (input_type='passage', measured payload batching, retry/backoff on 429/5xx,
           resumable ledger) and idempotently upsert to versioned Pinecone namespace.
4. verify: Full D1-D5 verification against Pinecone, idempotent second-run check, and DEV retrieval eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.data_pipeline import run_cleaning_and_snapshot  # noqa: E402
from backend.app.services.historical_stats import HistoricalStatsService  # noqa: E402

CURATED_PATH = PROJECT_ROOT / "data" / "disastermind_curated_dataset.json"
XLSX_PATH = (
    PROJECT_ROOT
    / "data"
    / "public_emdat_custom_request_2026-06-09_9221c56d-9867-40b3-9b51-cef53eda41e3.xlsx"
)
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "disastermind_snapshot.parquet"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "quarantine.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "snapshot_manifest.json"
TASK_2P_DIR = PROJECT_ROOT / "verification" / "task_2p"


def _write_reconciliation_reports(summary: dict[str, Any]) -> None:
    TASK_2P_DIR.mkdir(parents=True, exist_ok=True)
    rec = summary["reconciliation"]
    (TASK_2P_DIR / "reconciliation_report.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# Task 2P — Dataset Reconciliation Report (Curated JSON vs EM-DAT XLSX)",
        "",
        f"- **Curated JSON Path:** `{CURATED_PATH.name}` ({rec['curated_count']:,} rows, {rec['curated_unique_ids']:,} unique IDs)",
        f"- **EM-DAT XLSX Path:** `{XLSX_PATH.name}` ({rec['xlsx_count']:,} rows, {rec['xlsx_unique_ids']:,} unique IDs)",
        f"- **Shared IDs:** {rec['shared_id_count']:,}",
        f"- **Only in XLSX:** {rec['only_in_xlsx_count']}",
        f"- **Only in Curated JSON:** {rec['only_in_curated_count']}",
        f"- **Duplicate IDs in Curated JSON:** {len(rec['curated_duplicate_ids'])}",
        f"- **Duplicate IDs in XLSX:** {len(rec['xlsx_duplicate_ids'])}",
        f"- **Mapped Fields Compared:** {rec['mapped_fields_checked']}",
        f"- **Total Field Disagreements Across Shared Records:** {rec['total_field_disagreements']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Curated row count | {rec['curated_count']:,} |",
        f"| EM-DAT XLSX row count | {rec['xlsx_count']:,} |",
        f"| Unmatched XLSX rows | {rec['only_in_xlsx_count']} |",
        f"| Unmatched Curated rows | {rec['only_in_curated_count']} |",
        f"| Total field disagreements (25 fields) | {rec['total_field_disagreements']} |",
    ]
    (TASK_2P_DIR / "reconciliation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _write_data_quality_report(summary: dict[str, Any], stats_probe: dict[str, Any]) -> None:
    TASK_2P_DIR.mkdir(parents=True, exist_ok=True)
    st = summary["search_text_stats"]
    dd = summary["dedupe"]
    md = [
        "# Task 2P — Canonical Data Quality, Quarantine & Dedupe Report",
        "",
        f"- **Canonical `dataset_version` (SHA-256 prefix):** `{summary['dataset_version']}`",
        f"- **Raw Input Records:** {summary['raw_count']:,}",
        f"- **Canonical Snapshot Records:** {summary['cleaned_count']:,}",
        f"- **Quarantined Records:** {summary['quarantined_count']:,}",
        "",
        "## Quarantine Counts by Reason Code",
        "",
        "| Reason Code | Count |",
        "|---|---|",
    ]
    if summary["quarantine_counts_by_reason"]:
        for code, count in sorted(summary["quarantine_counts_by_reason"].items()):
            md.append(f"| `{code}` | {count} |")
    else:
        md.append("| *(none)* | 0 |")

    md.extend([
        "",
        "## Deduplication Summary",
        "",
        f"- Exact `disaster_id` duplicates quarantined: {dd['exact_id_duplicates']}",
        f"- Exact content-hash duplicates quarantined: {dd['exact_content_duplicates']}",
        f"- Near-duplicate same-(type, country, date) clusters (retained as distinct EM-DAT events): {dd['near_duplicate_same_day_country_type_clusters']}",
        "",
        "## Quality Flags Across Snapshot",
        "",
        "| Quality Flag | Record Count |",
        "|---|---|",
    ])
    for flag, count in sorted(summary["quality_flag_counts"].items()):
        md.append(f"| `{flag}` | {count:,} |")

    md.extend([
        "",
        "## Canonical `search_text` Bounds (Snapshot)",
        "",
        f"- **Max characters:** {st['max_chars']:,} (record `{st['max_chars_id']}`)",
        f"- **Max words:** {st['max_words']:,}",
        f"- **Max `cl100k_base` tokens:** {st['max_cl100k_tokens']:,} (record `{st['max_cl100k_tokens_id']}`)",
        f"- **Min characters / tokens:** {st['min_chars']} chars / {st['min_cl100k_tokens']} tokens",
        f"- **Records containing literal `'unknown'`:** {st['contains_unknown_literal']}",
        f"- **Records with empty `search_text`:** {st['empty_search_text']}",
        "",
        "## Historical Stats Service Smoke Verification (from Canonical Parquet Snapshot)",
        "",
        f"- **Query:** `disaster_type='Storm', country='India'` -> status=`{stats_probe['status']}`, sample_size=`{stats_probe['sample_size']}`",
        f"- **Total Deaths (median / p90):** {stats_probe['metrics'].get('total_deaths', {}).get('median')} / {stats_probe['metrics'].get('total_deaths', {}).get('p90')}",
        f"- **Total Affected (median / p90):** {stats_probe['metrics'].get('total_affected', {}).get('median')} / {stats_probe['metrics'].get('total_affected', {}).get('p90')}",
    ])
    (TASK_2P_DIR / "data_quality_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run_stage_clean() -> dict[str, Any]:
    summary = run_cleaning_and_snapshot(
        curated_path=CURATED_PATH,
        xlsx_path=XLSX_PATH,
        snapshot_path=SNAPSHOT_PATH,
        quarantine_path=QUARANTINE_PATH,
        manifest_path=MANIFEST_PATH,
    )
    _write_reconciliation_reports(summary)
    stats_svc = HistoricalStatsService(snapshot_path=SNAPSHOT_PATH)
    stats_probe = stats_svc.query_stats(disaster_type="Storm", country="India")
    _write_data_quality_report(summary, stats_probe)
    print(f"[CLEAN OK] dataset_version={summary['dataset_version']}")
    print(f"  Raw: {summary['raw_count']} | Clean: {summary['cleaned_count']} | Quarantined: {summary['quarantined_count']}")
    print(f"  Quarantine reasons: {summary['quarantine_counts_by_reason']}")
    print(f"  Snapshot max search_text: {summary['search_text_stats']['max_chars']} chars, {summary['search_text_stats']['max_words']} words, {summary['search_text_stats']['max_cl100k_tokens']} cl100k_base tokens")
    print(f"  'unknown' count in clean search_text: {summary['search_text_stats']['contains_unknown_literal']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="DisasterMind AI Task 2P Pipeline")
    parser.add_argument(
        "--stage",
        choices=["clean"],
        required=True,
        help="Pipeline stage to execute",
    )
    args = parser.parse_args()
    if args.stage == "clean":
        run_stage_clean()
    return 0


if __name__ == "__main__":
    sys.exit(main())
