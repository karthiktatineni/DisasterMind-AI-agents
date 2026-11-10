# Pre-Registered Serving Threshold Calibration Protocol (Task 2P — A2)

**Status:** Pre-registered and committed **prior** to embedding or scoring any of the 30 expanded DEV negative queries in `eval/dev_negatives_expanded.json`.  
**Model & Index (Locked):** `llama-text-embed-v2` (Pinecone-hosted, 1024-dim, namespace `v2-3d35e5131d79a79d-llama-text-embed-v2`, index `disastermind`). Model selection is NOT reopened.  
**Disclosure (R10):** Every metric calibrated or evaluated on `eval/dev_set.json` + `eval/dev_negatives_expanded.json` is **in-sample and optimistic** relative to unseen operational traffic.

---

## 1. Frozen DEV Negative Set (`35` Total Negatives)

- **Existing OOD DEV Queries (`eval/dev_set.json`, `DEV-RET-36..40`):** `5`
- **Additional DEV Negative Queries (`eval/dev_negatives_expanded.json`, `DEV-NEG-01..30`):** `30` (5 per category: `other_language`, `typos_only`, `vague`, `unsupported_hazard`, `wrong_region_or_era`, `non_disaster_text`).
- **Frozen SHA-256 of `eval/dev_negatives_expanded.json`:** `dc073da48fa015a81eac080481172b4286166cd1bb52bac86f4f0566fab21010`
- **Holdout Overlap Check (`eval/holdout_sealed.json`):** `exact_overlaps = 0`, `jaccard >= 0.70 overlaps = 0` (`25` holdout queries inspected for counts only; zero holdout text printed or scored).

---

## 2. Pre-Registered Threshold Selection Rule

1. Query live Pinecone namespace `v2-3d35e5131d79a79d-llama-text-embed-v2` (with `input_type="query"`, `top_k=10`, raw cosine similarity returned directly by Pinecone) for all **35 DEV negative queries** (`DEV-RET-36..40` + `DEV-NEG-01..30`). No changes to negative queries or model parameters are permitted after this commit.
2. Let $S_{\max} = \max_{q \in \text{DEV negatives}} \text{top1\_score}(q)$.
3. Set the calibrated serving reject threshold $\tau$ to the **lowest 4-decimal value strictly above $S_{\max}$** so that zero DEV negatives trigger a false positive (`0 / 35` FPs):
   $$\tau = \frac{\lfloor S_{\max} \times 10000 \rfloor + 1}{10000}$$
   *(Equivalently, with Pinecone scores rounded to 4 decimal places in `RetrievalResult`, $\tau = \text{round}(S_{\max}, 4) + 0.0001$, ensuring `top1_score >= threshold` is `False` for every negative in the 35-query DEV negative suite.)*
4. Evaluate the **35 answerable DEV queries** (`eval/dev_set.json`, `DEV-RET-01..35`) under BOTH protocols cleanly:
   - **Table A — Retrieval-Only (Unthresholded):** ranking quality ignoring $\tau$.
   - **Table B — End-to-End Serving ($\tau$-Enforced):** any answerable query whose `top1_score < \tau` returns `status="no_data"`, counts as an **abstention**, and scores **0** (`miss`, `RR = 0.0`) for `Recall@1/5/10` and `MRR`.
5. Report answerable rejection rate with 95% confidence interval (Wilson score & bootstrap CI).
