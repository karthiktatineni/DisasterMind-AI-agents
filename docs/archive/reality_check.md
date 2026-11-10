> [!NOTE]
> **Pre-Fix System Archive**: This document describes the historical pre-fix state of DisasterMind AI prior to the principal engineer / adversarial QA remediation overhaul. It is preserved strictly for audit trail and baseline comparison purposes.

# Reality Check — Evidence Pipeline Verification

## Checklist

| Step | Status | Notes |
|------|--------|-------|
| 1. Embeddings generated | PENDING | Run `python embeddings/generate_embeddings.py` with `NVIDIA_API_KEY` |
| 2. Uploaded to Supabase | PENDING | Run `python database/upload_embeddings.py` after schema deploy |
| 3. Stored in pgvector | PENDING | Apply `database/schema.sql` via Supabase SQL editor or MCP |
| 4. Retrieved through vector search | PENDING | Run `python database/verify_retrieval.py` |
| 5. Used by agents | IMPLEMENTED | Orchestrator calls `SimilarityService` + `RAGService` before agents |
| 6. Displayed on dashboard | IMPLEMENTED | Retrieved disasters + Evidence Used panels |

## Required Environment

```env
NVIDIA_API_KEY=...
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embed-v1
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
```

## Commands

```bash
pip install -r requirements.txt
python embeddings/generate_embeddings.py
# Apply database/schema.sql in Supabase
python database/upload_embeddings.py
python models/train_models.py
python database/verify_connection.py
python database/verify_retrieval.py
pytest tests/
```

## Architecture Proof

```
Dataset → embeddings/generate_embeddings.py → embedded_disasters.json
       → database/upload_embeddings.py → disaster_embeddings (pgvector)
       → backend/services/similarity_service.py → agents → dashboard
```

No local JSON search fallback is implemented in production services.
