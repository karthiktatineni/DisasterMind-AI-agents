-- DisasterMind AI - Production pgvector schema for NVIDIA nv-embed-v1
-- NVIDIA nv-embed-v1 returns 4096-dimensional vectors.

CREATE EXTENSION IF NOT EXISTS vector;

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
    embedding vector(4096) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS disaster_embeddings_disaster_type_idx
    ON disaster_embeddings (disaster_type);

CREATE INDEX IF NOT EXISTS disaster_embeddings_disaster_subtype_idx
    ON disaster_embeddings (disaster_subtype);

CREATE INDEX IF NOT EXISTS disaster_embeddings_country_idx
    ON disaster_embeddings (country);

CREATE INDEX IF NOT EXISTS disaster_embeddings_start_year_idx
    ON disaster_embeddings (start_year);

-- pgvector approximate indexes for the vector type are limited below 4096 dimensions.
-- This dataset is ~16k records, so exact cosine ordering is acceptable and keeps
-- NVIDIA vectors untruncated.

CREATE OR REPLACE FUNCTION match_disasters(
    query_embedding vector(4096),
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
    LIMIT LEAST(GREATEST(match_count, 1), 200);
$$;

GRANT EXECUTE ON FUNCTION match_disasters(vector, INTEGER, DOUBLE PRECISION, TEXT, TEXT)
    TO anon, authenticated, service_role;

ALTER TABLE disaster_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow read access for service role"
    ON disaster_embeddings
    FOR SELECT
    TO service_role
    USING (true);

CREATE POLICY "Allow read access for authenticated"
    ON disaster_embeddings
    FOR SELECT
    TO authenticated
    USING (true);

CREATE POLICY "Allow write access for service role"
    ON disaster_embeddings
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
