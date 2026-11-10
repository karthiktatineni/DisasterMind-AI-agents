# DisasterMind AI — Open Items (`verification/OPEN_ITEMS.md`)

Tracks external/non-code items surfaced during Tasks 0–2P that do not halt local execution unless explicitly escalated.

---

## 1. Raw EM-DAT Dataset Still Exposed on Public GitHub Remote (`origin/main`) — A11

- **Current Local State (`fix/adversarial-qa`):**
  - `git rev-list --objects fix/adversarial-qa | Select-String -Pattern "data/|embeddings/"` returns **only** code scripts (`embeddings/build_search_text.py`, `embeddings/generate_embeddings.py`). **Zero** licensed dataset/snapshot/quarantine/cache files exist on `fix/adversarial-qa` or local `main`.
- **Current Remote State (`origin/main` at `83dc5336e21387c5c2148313c9b8710f97e81109`):**
  - `origin/main` still serves **`19,329,771` bytes (`18.43 MiB`)**, of which `data/+embeddings/` accounts for **`18,856,980` bytes (`17.98 MiB`)**:
    - `data/disastermind_curated_dataset.json`: `12,641,170` bytes
    - `data/public_emdat_custom_request_2026-06-09_9221c56d-9867-40b3-9b51-cef53eda41e3.xlsx`: `6,207,869` bytes
    - `embeddings/build_search_text.py` + `embeddings/generate_embeddings.py`: `7,404` bytes
  - Live GitHub API inspection (`https://api.github.com/repos/karthiktatineni/DisasterMind-AI-agents`) reported: `private: False`, `visibility: "public"`, `forks_count: 0`, `size: 7187` KB.
- **Prepared (NOT Executed) Remediation Plan:**
  1. Immediately set `karthiktatineni/DisasterMind-AI-agents` visibility to **Private** in GitHub Repository Settings -> Danger Zone.
  2. Audit forks (`forks_count == 0` at check time), collaborator clones, GitHub Releases, and any deployment previews (Render/Vercel) to confirm no copies exist.
  3. Only with explicit repository owner approval under STOP CONDITION (a): force-push the already filtered local branches and tags (`git push origin --force main`, `git push origin --force fix/adversarial-qa`, `git push origin --force --tags`) so `origin` points strictly to rewritten commits free of `data/*.json` / `*.xlsx`.
  4. Contact GitHub Support (with commit SHA `83dc5336` / `6dcd3604`) to purge unreachable git objects and cached blob views from GitHub's servers before ever re-opening public visibility.

---

## 2. EM-DAT Terms of Use & Public Display Compliance — A7

- **Status:** **OPEN** (for owner / institutional legal review; documented verbatim in [`docs/DATA_LICENSE.md`](../docs/DATA_LICENSE.md)).
- **Open Questions:**
  - Whether hosting all 16,833 EM-DAT-derived metadata fields (`search_text`, impact totals, coordinates/locations) in Pinecone constitutes a restricted derivative database or internet distribution if surfaced via a public web API/demo.
  - Whether public demo display requires the UI/API minimization option proposed in `docs/DATA_LICENSE.md` Section 4 (serving only event name, year, country, type, and cohort aggregates with mandatory EM-DAT citation).

---

## 3. Pinecone Starter Monthly Limits (Inference Tokens & 1 GB Egress) — R12

- **Observed Usage & Dashboard Check (2026-11-10):**
  - Pinecone `/embed` HTTP response headers (`x-pinecone-api-version: 2025-04`) do **not** report account billing plan tier or monthly tokens remaining. Live 96-record sample measurement showed `13,476` Pinecone `usage.total_tokens` vs `6,996` `cl100k_base` tokens (`1.9262x` ratio), implying `~2.49M` inference tokens per 16,833-row `llama-text-embed-v2` pass (`~7–8M` cumulative tokens across prior Task 2P benchmarks and testing vs documented Starter plan 5M tokens/month free quota).
  - User inspected the live Pinecone dashboard on `2026-11-10`: `disastermind` index is on a monthly **1 GB egress limit** (`read access paused until December 1st, 2026 UTC` if exceeded), with the egress bar already past halfway (> ~55% consumed, driven by prior Task 2P D1-D5 full-namespace reads).
- **Quota Conservation Action in Force:** Keep retrieval at small `top_k` (`<= 5`), avoid redundant full-namespace fetches/re-embeds, and use a **stratified 2,000-record sample** (not full 16,833 re-embed) for the Task 12 clean-clone namespace check per R12 unless billing upgrade / quota increase is explicitly confirmed.

---

## 4. NVIDIA API Catalog Trial Terms — owner/legal review

- **Checked:** 2026-11-10.
- **Source:** [NVIDIA API Trial Terms of Service](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf), Section 1.4, page 1.
- **Relevant clause:** “you may only use the API Service for internal testing and evaluation purposes, not in production.”
- **Status:** OPEN. This records the source language only; no legal conclusion is made here. Confirm the applicable subscription and permitted deployment use with NVIDIA and the project owner before production use.
- **Rate limits:** Do not distribute requests among accounts to avoid trial quotas. The NVIDIA Developer Terms of Use, Sections 5(i) and 5(n), prohibit circumvention and using the service to avoid quotas: [NVIDIA Terms of Use](https://developer.nvidia.com/legal/terms).

