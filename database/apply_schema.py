#!/usr/bin/env python3
"""Apply the disaster_embeddings pgvector(4096) schema via direct Postgres."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.services.embedding_service import EMBEDDING_DIM  # noqa: E402


def _connect():
    import psycopg2

    settings = get_settings()
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        return psycopg2.connect(db_url)

    password = os.getenv("SUPABASE_DB_PASSWORD", "").strip()
    if not password:
        raise SystemExit("Set SUPABASE_DB_PASSWORD or DATABASE_URL in .env.")

    project_ref = settings.supabase_url.replace("https://", "").split(".")[0]
    host = f"db.{project_ref}.supabase.co"
    return psycopg2.connect(
        host=host,
        port=5432,
        dbname="postgres",
        user="postgres",
        password=password,
        sslmode="require",
    )


STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "DROP FUNCTION IF EXISTS match_disasters(vector, integer, text, text)",
    "DROP FUNCTION IF EXISTS match_disasters(vector, integer, double precision, text, text)",
    "DROP INDEX IF EXISTS disaster_embeddings_embedding_ivfflat_idx",
    f"""
    CREATE TABLE IF NOT EXISTS disaster_embeddings (
        id TEXT PRIMARY KEY,
        event_name TEXT,
        disaster_type TEXT,
        disaster_subtype TEXT,
        country TEXT,
        region TEXT,
        location TEXT,
        start_year INTEGER,
        total_deaths DOUBLE PRECISION,
        total_affected DOUBLE PRECISION,
        total_damage DOUBLE PRECISION,
        search_text TEXT NOT NULL,
        embedding vector({EMBEDDING_DIM}) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS event_name TEXT",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS disaster_type TEXT",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS disaster_subtype TEXT",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS country TEXT",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS region TEXT",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS location TEXT",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS start_year INTEGER",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS total_deaths DOUBLE PRECISION",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS total_affected DOUBLE PRECISION",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS total_damage DOUBLE PRECISION",
    "ALTER TABLE disaster_embeddings ADD COLUMN IF NOT EXISTS search_text TEXT",
    f"ALTER TABLE disaster_embeddings ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM})",
    "ALTER TABLE disaster_embeddings ALTER COLUMN search_text SET NOT NULL",
    "ALTER TABLE disaster_embeddings ALTER COLUMN embedding SET NOT NULL",
    "CREATE INDEX IF NOT EXISTS disaster_embeddings_disaster_type_idx ON disaster_embeddings (disaster_type)",
    "CREATE INDEX IF NOT EXISTS disaster_embeddings_disaster_subtype_idx ON disaster_embeddings (disaster_subtype)",
    "CREATE INDEX IF NOT EXISTS disaster_embeddings_country_idx ON disaster_embeddings (country)",
    "CREATE INDEX IF NOT EXISTS disaster_embeddings_start_year_idx ON disaster_embeddings (start_year)",
    f"""
    CREATE OR REPLACE FUNCTION match_disasters(
        query_embedding vector({EMBEDDING_DIM}),
        match_count INTEGER DEFAULT 10,
        match_threshold DOUBLE PRECISION DEFAULT 0.0,
        filter_disaster_type TEXT DEFAULT NULL,
        filter_country TEXT DEFAULT NULL
    )
    RETURNS TABLE (
        id TEXT,
        event_name TEXT,
        disaster_type TEXT,
        disaster_subtype TEXT,
        country TEXT,
        region TEXT,
        location TEXT,
        start_year INTEGER,
        total_deaths DOUBLE PRECISION,
        total_affected DOUBLE PRECISION,
        total_damage DOUBLE PRECISION,
        search_text TEXT,
        similarity DOUBLE PRECISION
    )
    LANGUAGE sql
    STABLE
    AS $$
        WITH ranked AS (
            SELECT
                de.id,
                de.event_name,
                de.disaster_type,
                de.disaster_subtype,
                de.country,
                de.region,
                de.location,
                de.start_year,
                de.total_deaths,
                de.total_affected,
                de.total_damage,
                de.search_text,
                (1 - (de.embedding <=> query_embedding))::DOUBLE PRECISION AS similarity,
                de.embedding <=> query_embedding AS distance
            FROM disaster_embeddings de
            WHERE
                (
                    filter_disaster_type IS NULL
                    OR de.disaster_type ILIKE '%' || filter_disaster_type || '%'
                    OR de.disaster_subtype ILIKE '%' || filter_disaster_type || '%'
                    OR de.search_text ILIKE '%' || filter_disaster_type || '%'
                )
                AND (
                    filter_country IS NULL
                    OR de.country ILIKE '%' || filter_country || '%'
                )
        )
        SELECT
            ranked.id,
            ranked.event_name,
            ranked.disaster_type,
            ranked.disaster_subtype,
            ranked.country,
            ranked.region,
            ranked.location,
            ranked.start_year,
            ranked.total_deaths,
            ranked.total_affected,
            ranked.total_damage,
            ranked.search_text,
            ranked.similarity
        FROM ranked
        WHERE ranked.similarity >= match_threshold
        ORDER BY ranked.distance
        LIMIT LEAST(GREATEST(match_count, 1), 200)
    $$
    """,
    "GRANT SELECT, INSERT, UPDATE ON disaster_embeddings TO service_role",
    "GRANT SELECT ON disaster_embeddings TO authenticated",
    "GRANT EXECUTE ON FUNCTION match_disasters(vector, integer, double precision, text, text) TO anon, authenticated, service_role",
    "ALTER TABLE disaster_embeddings ENABLE ROW LEVEL SECURITY",
    """
    DO $$ BEGIN
        CREATE POLICY "service_role_read" ON disaster_embeddings FOR SELECT TO service_role USING (true);
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$
    """,
    """
    DO $$ BEGIN
        CREATE POLICY "authenticated_read" ON disaster_embeddings FOR SELECT TO authenticated USING (true);
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$
    """,
    """
    DO $$ BEGIN
        CREATE POLICY "service_role_write" ON disaster_embeddings FOR ALL TO service_role USING (true) WITH CHECK (true);
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$
    """,
]


def apply() -> None:
    conn = _connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for statement in STATEMENTS:
                cur.execute(statement)
                print(f"OK: {statement.strip().split(chr(10))[0][:90]}")
    finally:
        conn.close()
    print(f"Schema applied (vector {EMBEDDING_DIM}).")


if __name__ == "__main__":
    apply()
