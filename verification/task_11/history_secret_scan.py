"""Scan reachable Git history for credential-shaped strings without printing values."""
from __future__ import annotations

import subprocess
import sys
import hashlib
import json
import re
from pathlib import Path

PATTERN = re.compile(r"nvapi-[A-Za-z0-9_-]{10,}|pcsk_[A-Za-z0-9_-]{10,}|sb_secret_[A-Za-z0-9_-]{10,}")


def run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def main() -> int:
    allowlist = json.loads(Path("config/secret_allowlist.json").read_text(encoding="utf-8"))
    allowed_hashes = {item["token_sha256"] for item in allowlist["allowed_hits"]}
    commits = run(["git", "rev-list", "--all"])
    if commits.returncode:
        print("HISTORY_SECRET_SCAN_ERROR: unable to list commits")
        return 2
    unexpected: list[tuple[str, str]] = []
    allowed_hits = 0
    for commit in commits.stdout.decode("ascii").splitlines():
        found = run(["git", "grep", "-I", "-n", "-E", PATTERN.pattern, commit, "--"])
        if found.returncode not in (0, 1):
            print("HISTORY_SECRET_SCAN_ERROR: git grep failed")
            return 2
        for line in found.stdout.decode("utf-8", "replace").splitlines():
            parts = line.split(":", 3)
            file_path = parts[1] if len(parts) > 2 else "<unparsed>"
            tokens = PATTERN.findall(parts[3] if len(parts) > 3 else "")
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                if digest in allowed_hashes:
                    allowed_hits += 1
                else:
                    unexpected.append((commit[:12], file_path))
    print(f"Commits scanned: {len(commits.stdout.decode('ascii').splitlines())}")
    print(f"Known synthetic negative-control token references (hash allowlisted): {allowed_hits}")
    if unexpected:
        print(f"Unexpected credential-shaped paths: {unexpected}")
        return 1
    print("HISTORY_SECRET_SCAN_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
