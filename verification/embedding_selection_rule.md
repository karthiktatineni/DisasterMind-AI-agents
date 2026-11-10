# Pre-Registered Embedding Model Selection Protocol (Task 1P / Task 2P)

**Status:** Pre-registered prior to running any candidate embedding comparison on `eval/dev_set.json`.  
**Seal:** `eval/holdout_sealed.json` (`SHA-256: ac086af95630a0626c80aff1c3a35e4f60017ea9a4ec7e37605b8d6393506285`) is NOT used in model selection.

---

## 1. Candidate Models

All candidates are verified against live provider model listings before benchmarking:

1. **NVIDIA NIM Candidates (`GET https://integrate.api.nvidia.com/v1/models`):**
   - `nvidia/nemotron-3.5-embed-1b` (dimension: `2048`, asymmetric `query`/`passage`)
   - `nvidia/nemotron-3-embed-1b` (dimension: `2048`, asymmetric `query`/`passage`)
   - `nvidia/nemotron-3.5-embed-8b` (dimension: `4096`, asymmetric `query`/`passage`)
2. **Pinecone-Hosted Candidates (`pc.inference.list_models()`):**
   - `llama-text-embed-v2` (dimension: `1024`, asymmetric `query`/`passage`)
   - `multilingual-e5-large` (dimension: `1024`, asymmetric `query`/`passage`)

---

## 2. Evaluation Set & Ground-Truth Matching

- **Dataset:** Canonical cleaned snapshot from Task 2P (embedded locally and cached per-model under `eval/.cache/<model_slug>/`, git-ignored).
- **Query Suite:** `eval/dev_set.json` (40 retrieval queries total):
  - **35 Answerable Queries** across 4 categories: `relevant` (20), `typo` (5), `vague` (5), `multilingual` (5).
  - **5 Out-of-Domain (OOD) Queries** (`out_of_domain`: 5).
- **Exact In-Memory Retrieval:** Exact L2-normalized cosine similarity (`dot(X, q)`) across all snapshot vectors in memory (`float32`), top-10 retrieved per query.
- **Relevance Definition:** A retrieved record matches an answerable DEV query if:
  - Its normalized `disaster_type` matches `query["type"]` (case-insensitive substring match against `disaster_type` or `disaster_subtype`), AND
  - If `query["country"]` is non-null, its normalized `country` matches `query["country"]` (case-insensitive exact/substring match).

---

## 3. Reported Metrics & Statistical Uncertainty

For every candidate model, record:
- **Recall@1, Recall@5, Recall@10** on the 35 answerable DEV queries.
- **Mean Reciprocal Rank (MRR@10)** on the 35 answerable DEV queries, with **95% bootstrap confidence intervals** (2,000 non-parametric bootstrap resamples over the 35 answerable queries, `seed=42`).
- **Paired Bootstrap Delta (`delta_MRR`):** 95% bootstrap CI on `MRR(Model A) - MRR(Model B)` using identical query resamples.
- **OOD False-Positive Rate:** Number of the 5 OOD queries whose top-1 cosine similarity exceeds the candidate's DEV-calibrated similarity reject threshold (thresholds calibrated on DEV to separate answerable vs OOD queries, recorded alongside ROC/separation margin).
- **Mean Query Embedding Latency (ms)** and **Vector Dimension**.

---

## 4. Methodological Caveat (Small Sample Size)

The DEV retrieval suite contains **35 answerable queries** (and 5 OOD queries), comparable in scale to the pre-fix 21-query domain-relevant suite. With $n = 35$, small MRR differences (e.g. $0.01\text{–}0.04$) correspond to 1–2 queries shifting rank and are statistically indistinguishable from noise unless confirmed by paired bootstrap intervals.

---

## 5. Pre-Registered Winner & Tie-Breaker Rule

Selection proceeds strictly in the following order:

1. **Minimum Bar Gate (Must Pass to Be Eligible):**
   - OOD false-positive count at the model's calibrated threshold is **no worse than the pre-fix baseline** ($\le 2/5$ OOD false positives; target $0/5$), AND
   - Answerable MRR is **$\ge 0.6012$** (or if no candidate reaches `0.6012` on the harder 35-query DEV set containing typos/vague/multilingual queries, the candidate with the highest answerable MRR that satisfies the OOD false-positive bar).
2. **Primary Ranking Criterion:**
   - Rank eligible models by **MRR@10 on the 35 answerable DEV queries** (highest first).
3. **Statistical Significance vs. Tie Criterion:**
   - Let $M_1$ be the top-MRR eligible model and $M_2$ be the runner-up (or any other eligible model).
   - $M_1$ **strictly wins** over $M_2$ if and only if the **95% paired bootstrap CI of $\text{MRR}(M_1) - \text{MRR}(M_2)$ is strictly $> 0$** (lower bound $> 0.0$).
   - If the 95% paired bootstrap CI of $\text{MRR}(M_1) - \text{MRR}(M_2)$ **includes $0$** (equivalently, individual bootstrap CIs overlap without statistically significant separation), **declare a statistical tie** among all models not significantly worse than $M_1$.
4. **Deterministic Tie-Breaker (Applied Among Tied Top Models):**
   1. Prefer **existing Pinecone index dimension (`2048`)** if difference in storage/latency is comparable, OR **smaller vector dimension** (lower storage & RU cost: `1024 <= 2048 < 4096`) when re-indexing is trivial/equal — specifically:
      - If a `1024`-dim or `2048`-dim model is tied with a `4096`-dim model, reject `4096`-dim (half storage/latency/memory efficiency).
      - Between tied `1024`-dim and `2048`-dim models (or among tied `2048`-dim models): prefer **lower dimension** first (`1024` over `2048`), or if same dimension (`2048`), prefer **lower mean query latency** (and if latencies are within 10%, prefer the newer `nvidia/nemotron-3.5-embed-1b` over `nvidia/nemotron-3-embed-1b`).
   - *Note on index dimension:* If the winning model's dimension differs from the current `2048`-dim `disastermind` index, recreate the Pinecone index with the winning dimension and update `.env`, `backend/app/config.py`, `scripts/preflight.py`, and docs before ingestion.
   - *Note on provider branding:* Retain "NVIDIA embeddings" claims in documentation/UI only if an NVIDIA model wins.
