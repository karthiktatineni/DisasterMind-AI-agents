# EM-DAT Data License, Provenance, Project Exposure & Open Items

## 1. Source Dataset Provenance (`EM-DAT Info` Sheet)

From `data/public_emdat_custom_request_2026-06-09_9221c56d-9867-40b3-9b51-cef53eda41e3.xlsx` (`EM-DAT Info` worksheet):

```text
Source:        EM-DAT, CRED / UCLouvain, Brussels, Belgium
               https://www.emdat.be
Glossary:      https://doc.emdat.be/docs/data-structure-and-content/emdat-public-table/
Version:       2026-05-20
File creation: Tue, 09 Jun 2026 04:00:01 UTC
Table type:    public_emdat_custom_request
# of records:  16834
```

---

## 2. Terms of Use (Quoted Verbatim from Primary Sources)

- **Terms of Use URL:** `https://doc.emdat.be/docs/legal/terms-of-use/` (UCLouvain 2025; retrieved `2026-11-10`):

> "If you are an academic organization, a university, a non-profit research institution or an international public organization (UN agencies, multi-lateral banks, other multi-lateral institutions, and national governments) or part of a media agency (journalist, or press agency) with the intention to use the EM-DAT database (hereafter EM-DAT) for research, teaching or information purposes, you shall, conditional upon the acceptance of the present conditions of use, be granted free access to EM-DAT (see also Authorized Use)."
>
> "Users of EM-DAT declare that they will not (see also Unauthorized Use):
> - Reproduce, copy, communicate, lend, or otherwise distribute EM-DAT or a substantial part of EM-DAT.
> - Reverse assemble, decompile, decompose, or disassemble EM-DAT.
> - Create substitute or derivative databases of EM-DAT.
> - Attempt to unlock or bypass any initialization or security systems used by EM-DAT.
> - Share, use or transmit any portion of EM-DAT via the Internet to unauthorized users.
> - Divulge the password to unauthorized users.
> - Remove, alter or obscure any proprietary legend, copyright, trademark, or other intellectual property right notice, logo, and image in or on EM-DAT.
> - Perform acts that conflict with the normal exploitation of EM-DAT or unreasonably prejudice the interest of the Licensor.
> - Make Commercial Use of EM-DAT, except in the event such users enter into a separate Database License Agreement, described hereafter."
>
> "Except if agreed upon in a separate Database License Agreement, EM-DAT (including its data and derivate products) cannot be used for any commercial purpose."

- **Citation Policy URL:** `https://doc.emdat.be/docs/legal/citation-policy/` (retrieved `2026-11-10`):

> "All published materials using or referring to the EM-DAT data or the EM-DAT project or a part of it shall refer to it following the Proper Citation."
>
> **Comprehensive Citation Format:**  
> `Emergency Events Database (EM-DAT), UCLouvain / CRED, Brussels, Belgium, accessed on 2026-06-09, https://www.emdat.be/`
>
> **Minimal Citation Format:**  
> `EM-DAT data, 2026, www.emdat.be`

---

## 3. Project EM-DAT Exposure Inventory

1. **GitHub Remote (`origin/main`, public visibility verified at check time):**
   Still contains `data/disastermind_curated_dataset.json` (`12,641,170` bytes) and `data/public_emdat_custom_request_2026-06-09_9221c56d-9867-40b3-9b51-cef53eda41e3.xlsx` (`6,207,869` bytes). Tracked as Open Item #1 in [`verification/OPEN_ITEMS.md`](../verification/OPEN_ITEMS.md). Local branch `fix/adversarial-qa` and working-tree gitignores (`data/`, `embeddings/*.json`, `eval/.cache/`) exclude all raw/snapshot datasets.
2. **Pinecone Index (`disastermind`, namespace `v2-3d35e5131d79a79d-llama-text-embed-v2`):**
   Stores 16,833 vector embeddings plus metadata fields (`disaster_id`, `disaster_type`, `disaster_subtype`, `category_tag`, `event_name`, `display_label`, `country`, `region`, `location`, `start_year`, `total_deaths`, `total_affected`, `total_damage`, `search_text`). Because `search_text` and full metadata mirror cleaned EM-DAT rows, Pinecone acts as a private vector store holding a derivative representation.
3. **Backend API Responses (`backend/app/main.py` `/orchestrate`, `/orchestrate-intake`, `/agent-answer` & `rag_service.py` / `similarity_service.py`):**
   Return up to top-k matched precedents in `retrieved_disasters` and `evidence_used` (`id`, `event_name`, `disaster_type`, `disaster_subtype`, `country`, `region`, `location`, `start_year`, `total_deaths`, `total_affected`, `total_damage`, `similarity`) plus narrative/agent context.
4. **Frontend UI Tables (`frontend/app/page.tsx` lines 608–631, 754–777, 807–809):**
   Displays `start_year`, `location`/`region`, `country`, `total_affected`, `total_damage`, `disaster_type` and raw JSON `result.output` accordions if opened.
5. **Public Demo / External Hosting:**
   If deployed publicly without access gating, anyone could repeatedly query or scrape retrieved precedent fields.

---

## 4. Open Licensing Questions (For Project Owner Decision — Not Decided Here)

- **OPEN — Redistribution / Hosting Mirror (`origin/main` & Pinecone):** Does storing full `search_text` and per-event EM-DAT impact fields for all 16,833 rows in Pinecone constitute a *"substitute or derivative database"* or *"transmit any portion of EM-DAT via the Internet to unauthorized users"* if exposed outside private academic use?
- **OPEN — Public Demo / Display:** Would a public web UI returning top-10 historical event details per query fall within *"information purposes"* with proper citation or violate *"communicate, lend, or otherwise distribute ... via the Internet to unauthorized users"*?
- **OPEN — Commercial Use:** Any non-academic / commercial deployment is prohibited without a separate UCLouvain/CRED Database License Agreement (`regina.below@uclouvain.be`).

### Proposed Data Minimization Option (Before Any Public Deployment)
- Strip `search_text`, `location`, `latitude`/`longitude`, and raw disaggregated casualty/damage sub-fields from Pinecone metadata and public API responses; store only minimal retrieval keys (`disaster_id`, `event_name`, `start_year`, `country`, `disaster_type`, `category_tag`) alongside pre-computed cohort aggregates (`historical_stats` median / p90 with `n >= 5`) while rendering EM-DAT mandatory attribution in UI and reports.

---

## 5. Repository Hygiene Rules

1. Raw EM-DAT exports (`data/*.json`, `data/*.xlsx`), quarantine files (`data/quarantine.json`), canonical snapshots (`data/*.parquet`), local ledgers (`data/*.json`), and local embedding caches (`embeddings/*.json`, `eval/.cache/`) remain `.gitignore`d and never committed on local branches.
2. Any briefing or UI displaying historical precedent statistics must cite EM-DAT, CRED / UCLouvain (`www.emdat.be`).
