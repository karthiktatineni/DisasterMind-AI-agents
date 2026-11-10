> [!NOTE]
> **Pre-Fix System Archive**: This document describes the historical pre-fix state of DisasterMind AI prior to the principal engineer / adversarial QA remediation overhaul. It is preserved strictly for audit trail and baseline comparison purposes.

# pgvector Status

**Status:** Schema defined in `database/schema.sql` — requires deployment to Supabase.

## Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Table

- `disaster_embeddings` with `embedding vector(4096)` for `nvidia/nv-embed-v1`

## Index

- IVFFLAT cosine index on `embedding`

## Function

- `match_disasters(query_embedding, match_count, filter_disaster_type, filter_country)`

Run `python database/verify_connection.py` after configuring credentials to verify live status.
