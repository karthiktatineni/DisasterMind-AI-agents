# DisasterMind AI — System & Data Invariants (I1–I10, D1–D5)

## System Invariants (I1–I10)
- **I1:** No unhandled exception or raw 500 from any upstream failure (LLM, embeddings, DB/Pinecone, weather, geocoding).
- **I2:** Zero ungrounded numbers in any final briefing, recomputed by an independent checker from raw logs.
- **I3:** Zero invalid or unsupported citations.
- **I4:** Every evacuation recommendation carries `requires_human_approval = true`, set by code.
- **I5:** Zero physically impossible scenarios accepted (Category 6 cyclone, magnitude 11, lat 200, etc.).
- **I6:** When any tool fails or returns no data, zero fabricated substitute data. The agent abstains.
- **I7:** Zero secrets in the working tree (tracked files), git history, or the client bundle.
- **I8:** The UI never shows an agent as running or complete unless a real backend event says so.
- **I9:** Every briefing is labeled as a scenario planning aid, with historical records distinguished from the hypothetical scenario.
- **I10:** The event loop is never blocked more than 100 ms under 20 concurrent requests.

## Data & Pinecone Invariants (D1–D5)
- **D1:** Every cleaned record exists in the Pinecone namespace exactly once: id set equal to the snapshot, vector dimension correct, finite values, metadata equal to the snapshot.
- **D2:** No record with empty text, text containing the filler literal `"unknown"`, or duplicate content reaches the index.
- **D3:** Re-running ingestion changes nothing (idempotent): zero embedding calls and zero upserts on a second run.
- **D4:** The app refuses to start if the snapshot's `dataset_version`, the configured embedding model, or the namespace do not match.
- **D5:** No retrieval result is returned without its metadata. `historical_stats` returns `no_data` when the sample size is below a configured minimum ($n < 5$).
