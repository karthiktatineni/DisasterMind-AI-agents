"""Task gate runner (R7, R8, A6, A10): pure-Python offline test, grep, secret-allowlist, hygiene & holdout checks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = ROOT / "config" / "secret_allowlist.json"
EXPECTED_HOLDOUT_SHA256 = "ac086af95630a0626c80aff1c3a35e4f60017ea9a4ec7e37605b8d6393506285"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def step_offline_pytest() -> bool:
    print("=== 1. OFFLINE PYTEST SUITE ===", flush=True)
    proc = _run([sys.executable, "-m", "pytest", "-v"])
    print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip())
    print(f"STEP_EXIT={proc.returncode}\n", flush=True)
    return proc.returncode == 0


def step_grep_gates() -> bool:
    print("=== 2. SUPABASE / PGVECTOR / NVIDIA_EMBEDDING / STALE CLAIMS GREP GATES ===", flush=True)
    ok = True
    checks = [
        (
            "supabase_pgvector",
            ["git", "grep", "-n", "-i", "-E", r"supabase|pgvector", "--", ":!docs/archive/*", ":!verification/*", ":!scripts/run_gates.py"],
        ),
        (
            "nvidia_embedding_var",
            ["git", "grep", "-n", "-i", "NVIDIA_EMBEDDING", "--", ":!docs/archive/*", ":!verification/*", ":!scripts/run_gates.py"],
        ),
        (
            "stale_claims_docs_ui",
            ["git", "grep", "-n", "-i", "-E", r"nvidia.?embeddings|2048", "--", "readme.md", "docs", "frontend", ".env.example", "config", ":!docs/archive/*", ":!verification/*"],
        ),
    ]
    for name, cmd in checks:
        proc = _run(cmd)
        # git grep returns 1 when 0 matches found, 0 when matches exist, >1 on error
        matches = proc.stdout.strip()
        if proc.returncode == 0 or matches:
            print(f"[FAIL] {name} matched:\n{matches}")
            ok = False
        elif proc.returncode not in (1,):
            print(f"[FAIL] {name} grep errored (exit={proc.returncode}): {proc.stderr.strip()}")
            ok = False
        else:
            print(f"[PASS] {name}: 0 matches (git grep exit=1)")

    # Check untracked docs/DATA_LICENSE.md and local .env keys without printing .env secrets
    lic = ROOT / "docs" / "DATA_LICENSE.md"
    if lic.exists():
        text = lic.read_text(encoding="utf-8")
        if re.search(r"NVIDIA_EMBEDDING|nvidia.?embeddings|\b2048\b", text, re.IGNORECASE):
            print("[FAIL] docs/DATA_LICENSE.md contains prohibited embedding/2048 string")
            ok = False

    env_path = ROOT / ".env"
    if env_path.exists():
        bad_keys = [
            ln.split("=", 1)[0].strip()
            for ln in env_path.read_text(encoding="utf-8").splitlines()
            if "=" in ln and not ln.lstrip().startswith("#") and "NVIDIA_EMBEDDING" in ln.split("=", 1)[0].upper()
        ]
        if bad_keys:
            print(f"[FAIL] .env still contains legacy key(s): {bad_keys}")
            ok = False
        else:
            print("[PASS] local .env: 0 NVIDIA_EMBEDDING* keys")

    print(f"GREP_GATES_PASS={ok}\n", flush=True)
    return ok


def step_secret_scan() -> bool:
    print("=== 3. SECRET SCAN WITH ALLOWLIST ===", flush=True)
    allow_data = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))["allowed_hits"]
    allowed = {(item["path"], int(item["line"])): item["token_sha256"] for item in allow_data}
    pattern = r"nvapi-[A-Za-z0-9_-]{10,}|pcsk_[A-Za-z0-9_-]{10,}|sb_secret_[A-Za-z0-9_-]{10,}"

    proc = _run(["git", "grep", "-n", "-E", pattern])
    if proc.returncode > 1:
        print(f"[FAIL] secret grep error: {proc.stderr.strip()}")
        return False

    unexpected: list[str] = []
    allowlisted: list[str] = []
    for raw in proc.stdout.splitlines():
        parts = raw.split(":", 2)
        if len(parts) < 3:
            unexpected.append(raw)
            continue
        rel_path, line_str, content = parts[0], parts[1], parts[2]
        line_no = int(line_str)
        expected_sha = allowed.get((rel_path, line_no))
        tokens = [m.group(0) for m in re.finditer(pattern, content)]
        if expected_sha and tokens and all(hashlib.sha256(tok.encode("utf-8")).hexdigest() == expected_sha for tok in tokens):
            allowlisted.append(raw)
        else:
            unexpected.append(raw)

    for a in allowlisted:
        print(f"[ALLOWLISTED] {a}")
    for u in unexpected:
        print(f"[UNEXPECTED SECRET HIT] {u}")

    ok = len(unexpected) == 0
    print(f"SECRET_SCAN_EXIT={0 if ok else 1} (allowlisted={len(allowlisted)}, unexpected={len(unexpected)})\n", flush=True)
    return ok


def step_git_hygiene() -> bool:
    print("=== 4. GIT STATUS & TRACKED SENSITIVE-PATH HYGIENE ===", flush=True)
    forbidden_prefixes = ("data/", "eval/.cache/", ".env", ".cursor/")
    ls_proc = _run(["git", "ls-files"])
    st_proc = _run(["git", "status", "--porcelain"])
    bad_tracked = [p for p in ls_proc.stdout.splitlines() if p.startswith(forbidden_prefixes) and p != ".env.example"]
    bad_staged = [
        ln[3:].strip()
        for ln in st_proc.stdout.splitlines()
        if ln and ln[0] in ("A", "M", "R", "C") and ln[3:].strip().startswith(forbidden_prefixes)
    ]
    print(f"bad_tracked={bad_tracked}, bad_staged={bad_staged}")
    ok = not bad_tracked and not bad_staged
    print(f"GIT_HYGIENE_PASS={ok}\n", flush=True)
    return ok


def step_holdout_hash() -> bool:
    print("=== 5. SEALED HOLDOUT HASH VERIFICATION (HASHLIB) ===", flush=True)
    work_bytes = (ROOT / "eval" / "holdout_sealed.json").read_bytes()
    git_proc = subprocess.run(
        ["git", "show", "holdout-sealed:eval/holdout_sealed.json"],
        cwd=ROOT,
        capture_output=True,
    )
    if git_proc.returncode != 0:
        print(f"[FAIL] git show holdout-sealed failed: {git_proc.stderr.decode('utf-8', 'replace')}")
        return False

    work_sha = hashlib.sha256(work_bytes).hexdigest()
    tag_sha = hashlib.sha256(git_proc.stdout).hexdigest()
    ok = work_sha == EXPECTED_HOLDOUT_SHA256 and tag_sha == EXPECTED_HOLDOUT_SHA256
    print(f"holdout-sealed tag SHA-256 = {tag_sha}")
    print(f"working file SHA-256       = {work_sha}")
    print(f"expected SHA-256           = {EXPECTED_HOLDOUT_SHA256}")
    print(f"HOLDOUT_HASH_PASS={ok}\n", flush=True)
    return ok


def main() -> int:
    results = {
        "offline_pytest": step_offline_pytest(),
        "grep_gates": step_grep_gates(),
        "secret_scan": step_secret_scan(),
        "git_hygiene": step_git_hygiene(),
        "holdout_hash": step_holdout_hash(),
    }
    all_ok = all(results.values())
    print("=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"FINAL_EXIT={0 if all_ok else 1}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

