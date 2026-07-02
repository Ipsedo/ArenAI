---
name: setup-venv
description: Create a Python virtualenv (.venv) and install pre-commit if it doesn't already exist. Use when the user wants to bootstrap the Python environment, set up pre-commit, or asks to create the venv.
---

# Setup venv + pre-commit

Bootstrap a local Python virtualenv named `.venv` and install `pre-commit` into it, idempotently.

## Steps

Run from the directory where the venv should live (default: the current working directory).

1. **Create the venv only if missing.** Do not recreate an existing one:
   ```shell
   [ -d .venv ] || python -m venv .venv
   ```

2. **Install / upgrade pre-commit** in the venv:
   ```shell
   ./.venv/bin/python -m pip install --upgrade pip pre-commit
   ```

3. **Install the git hooks** if a `.pre-commit-config.yaml` exists (search current dir, then the repo root):
   ```shell
   [ -f .pre-commit-config.yaml ] && ./.venv/bin/pre-commit install
   ```

4. **Report** what was done: whether the venv was newly created or already present, the installed `pre-commit` version (`./.venv/bin/pre-commit --version`), and whether hooks were installed.

## Notes

- Idempotent by design — safe to re-run. An existing `.venv` is left untouched (step 1 skips creation; steps 2–3 just re-assert the desired state).
- Use `./.venv/bin/python` / `./.venv/bin/pre-commit` directly rather than `source`-ing the activate script, so it works non-interactively.
- On Windows the paths are `.venv\Scripts\python.exe` and `.venv\Scripts\pre-commit.exe`.
- In this repo the `.pre-commit-config.yaml` for the C++ code lives in the parent `PhyVR/` directory; the Python tooling has its own under `arenai_train/python/`. Pick the one matching the directory you're bootstrapping.
