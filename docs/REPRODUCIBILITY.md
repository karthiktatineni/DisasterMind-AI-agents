# Reproducible Local Setup

The checked Python runtime is pinned in `.python-version`; the fully resolved Python dependency set is `requirements.lock`. `requirements.txt` remains the editable direct-dependency list used to refresh that lock.

## Windows

From the repository root:

```powershell
uv venv .venv --python 3.14.6
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
Copy-Item .env.example .env
npm ci --prefix frontend
npx --prefix frontend playwright install chromium
npm run build --prefix frontend
.venv\Scripts\python.exe scripts\run_gates.py
```

Set provider credentials only in the ignored local `.env` and `frontend/.env.local` files. Never commit those files. The frontend uses the server-side `DISASTERMIND_API_KEY`; it is not exposed as a `NEXT_PUBLIC_*` variable.

To regenerate the Python lock after editing `requirements.txt`:

```powershell
uv pip compile requirements.txt --python-version 3.14 -o requirements.lock --no-annotate
```

The security gate is `python scripts/run_gates.py`. Dependency audits are `python -m pip_audit -r requirements.lock` and `npm audit --prefix frontend`.
