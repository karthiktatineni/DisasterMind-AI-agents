-- DisasterMind AI — Production pgvector schema
-- Requires: PostgreSQL with pgvector extension (Supabase)

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
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS disaster_embeddings_disaster_type_idx
    ON disaster_embeddings (disaster_type);

CREATE INDEX IF NOT EXISTS disaster_embeddings_country_idx
    ON disaster_embeddings (country);

CREATE INDEX IF NOT EXISTS disaster_embeddings_start_year_idx
    ON disaster_embeddings (start_year);

-- IVFFLAT index: build after bulk load for best recall
-- Run: SET ivfflat.probes = 10; before queries for tuning
CREATE INDEX IF NOT EXISTS disaster_embeddings_embedding_ivfflat_idx
    ON disaster_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Cosine similarity search (similarity = 1 - cosine distance)
CREATE OR REPLACE FUNCTION match_disasters(
    query_embedding vector(1536),
    match_count INTEGER DEFAULT 10,
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
        (1 - (de.embedding <=> query_embedding))::DOUBLE PRECISION AS similarity
    FROM disaster_embeddings de
    WHERE
        (filter_disaster_type IS NULL OR de.disaster_type ILIKE filter_disaster_type)
        AND (filter_country IS NULL OR de.country ILIKE filter_country)
    ORDER BY de.embedding <=> query_embedding
    LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION match_disasters(vector, INTEGER, TEXT, TEXT) TO anon, authenticated, service_role;

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
