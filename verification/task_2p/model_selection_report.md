# Task 2P Step 8 — Embedding Model Selection Report (DEV Set Exact Cosine)

- **Snapshot Version:** `3d35e5131d79a79d` (16,833 records)
- **DEV Suite:** 35 answerable queries + 5 OOD queries
- **Winner (per pre-registered rule):** `llama-text-embed-v2` (pinecone, 1024-dim)
- **Requires Index Recreation (`dim != 2048`):** True

| Model | Provider | Dim | R@1 | R@5 | R@10 | MRR (95% CI) | Calib. Threshold | OOD FP | Mean Query Latency |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| `nvidia/nemotron-3.5-embed-1b` | nvidia | 2048 | 0.8571 | 0.9143 | 0.9429 | **0.8857** [0.7810, 0.9714] | 0.3067 | 0/5 | 1753.8 ms |
| `nvidia/nemotron-3-embed-1b` | nvidia | 2048 | 0.8571 | 0.9429 | 0.9429 | **0.8867** [0.7819, 0.9714] | 0.3065 | 0/5 | 280.1 ms |
| `nvidia/nemotron-3.5-embed-8b` | nvidia | 4096 | 0.8571 | 0.8857 | 0.8857 | **0.8714** [0.7571, 0.9714] | 0.2910 | 0/5 | 301.3 ms |
| `llama-text-embed-v2` | pinecone | 1024 | 0.8571 | 0.9143 | 0.9429 | **0.8905** [0.7857, 0.9714] | 0.3211 | 0/5 | 391.3 ms |
| `multilingual-e5-large` | pinecone | 1024 | 0.6857 | 0.9429 | 0.9714 | **0.7948** [0.6781, 0.8971] | 0.8351 | 0/5 | 411.7 ms |

## Pairwise Paired Bootstrap Deltas vs. Top-MRR Model

- `llama-text-embed-v2` vs `nvidia/nemotron-3-embed-1b`: delta_MRR=+0.0038, 95% CI=[-0.0581, 0.0695], significant=False
- `llama-text-embed-v2` vs `nvidia/nemotron-3.5-embed-1b`: delta_MRR=+0.0048, 95% CI=[-0.0571, 0.0714], significant=False
- `llama-text-embed-v2` vs `nvidia/nemotron-3.5-embed-8b`: delta_MRR=+0.0190, 95% CI=[-0.0524, 0.1], significant=False
- `llama-text-embed-v2` vs `multilingual-e5-large`: delta_MRR=+0.0957, 95% CI=[0.0114, 0.1857], significant=True
