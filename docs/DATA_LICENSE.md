# EM-DAT Data License, Provenance & Usage Constraints

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

## 2. Terms of Use (Quoted from Primary Source)

Fetched directly from `https://doc.emdat.be/docs/legal/terms-of-use/` (UCLouvain 2025):

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

---

## 3. Mandatory Attribution & Citation Policy

Fetched directly from `https://doc.emdat.be/docs/legal/citation-policy/`:

> "All published materials using or referring to the EM-DAT data or the EM-DAT project or a part of it shall refer to it following the Proper Citation."
>
> **Comprehensive Citation Format:**
> `Emergency Events Database (EM-DAT), UCLouvain / CRED, Brussels, Belgium, accessed on 2026-06-09, https://www.emdat.be/`
>
> **Minimal Citation Format:**
> `EM-DAT data, 2026, www.emdat.be`

---

## 4. Repository Compliance Rules

1. **No Raw Redistribution in Git:** Raw EM-DAT exports (`data/*.json`, `data/*.xlsx`), quarantine files (`data/quarantine.json`), canonical snapshots (`data/*.parquet`), and local embedding caches (`embeddings/*.json`, `eval/.cache/`) are strictly `.gitignore`d and must never be committed or pushed to public repositories.
2. **Non-Commercial Research Scope:** DisasterMind AI is a non-commercial research and scenario-planning prototype. Commercial deployment requires a separate Database License Agreement with CRED / UCLouvain (`regina.below@uclouvain.be`).
3. **Attribution in Outputs & Docs:** Any report or UI view displaying historical EM-DAT precedents must cite EM-DAT, CRED / UCLouvain (`www.emdat.be`).

