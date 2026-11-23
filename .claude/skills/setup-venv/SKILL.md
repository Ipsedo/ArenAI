---
name: setup-venv
description: Create the Python virtualenv (.venv) at the repo root and install pre-commit if missing. Run when `./.venv/bin/pre-commit` is absent or a pre-commit command fails for lack of it.
---

# Setup venv + pre-commit

Idempotent. From the repo root:

```shell
[ -d .venv ] || python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip pre-commit
[ -f .pre-commit-config.yaml ] && ./.venv/bin/pre-commit install
./.venv/bin/pre-commit --version
```

- Never recreate an existing `.venv`.
- Call `./.venv/bin/…` directly; never `source` the activate script (non-interactive shell).
- Windows: `.venv\Scripts\python.exe`, `.venv\Scripts\pre-commit.exe`.
- The Python tooling under `arenai_agent/python/` has its own environment and requirements;
  this venv is for the C++ pre-commit pipeline only.
- Report: venv created or already present, pre-commit version, hooks installed or not.
