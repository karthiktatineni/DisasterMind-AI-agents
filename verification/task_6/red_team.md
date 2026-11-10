# Task 6 Red-Team & Evidence Self-Check (`8e6e1b3`)

1. **Code changed since live run (`8e6e1b3`):** Checked `git diff --name-only 8e6e1b3 -- backend tests config` -> `""` (`PASS=True`).
2. **Scripted path masking live inference:** Confirmed `62` live NIM calls across `4` full 4-agent runs + `8` live Risk runs against `nvidia/nemotron-3-ultra-550b-a55b` alongside labeled offline `tests/test_task6_injection.py` (`PASS=True`).
3. **Wrong namespace / throwaway cleanup:** Live `describe_index_stats()` confirmed `remaining_throwaway_namespaces=[]` and live namespace `v2-3d35e5131d79a79d-llama-text-embed-v2` untouched at `16833` vectors (`PASS=True`).
4. **Cached/reused responses:** Verified `12` distinct raw trace files produced `102` distinct server tool `call_id`s and `0` `RAW_SEARCH_TEXT_POISON_MARKER` leaks (`raw_marker_leak=False`, `PASS=True`).
5. **App-written log vs independent checks:** Re-ran `pytest tests/test_task6_injection.py -v` (`12 passed`) and `scripts/run_gates.py` (`PASS=True`).
