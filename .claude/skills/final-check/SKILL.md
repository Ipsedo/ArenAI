---
name: final-check
description: Final verification of code modifications via `pre-commit run --all-files` — formats (clang-format, whitespace), builds the project and runs every module's tests. ALWAYS the last step after modifying C++ code, before declaring the work done or committing (« vérifie », « final check », « lance pre-commit », « assure-toi que tout passe »).
---

# Final check

Hooks (`.pre-commit-config.yaml` at the repo root, `fail_fast: true`), in order:
`end-of-file-fixer`, `trailing-whitespace`, `clang-format`, `build` (`./build_linux.sh`),
`tests` (`ctest --build-dir ./build --output-on-failure`).

## 0. Self-review first

The pipeline checks that the code compiles and passes; it does not check that it is right.
Before running it, re-read the diff against `CLAUDE.md`:

- no shared surface changed without a validated question (`design-gate` §5 checklist: getters,
  threaded parameters, usage leaks, flags/defaults);
- no comment longer than one line, none that paraphrases or justifies;
- nothing unrequested: no extra option, default, feature, refactor.

## 1. Run

From the repo root; `pre-commit` lives in `./.venv` (missing ⟹ skill `setup-venv`):

```shell
git add -A <files you changed>
./.venv/bin/pre-commit run --all-files
```

New files must be staged, otherwise pre-commit does not see them. Bash timeout ≥ 10 min (cold
build).

## 2. Read failures

- **Formatting hooks** report `Failed` because they *fixed* files in place. Re-run; the second
  pass is green. Their fixes are part of the change: keep and re-stage them.
- **`build` / `tests`**: a real failure. Read the error, fix the cause, re-run from step 1.
- **Golden-image tests** (`arenai_view`, `arenai_core`) compare against lavapipe/Ubuntu
  references and may fail on the local GPU for that reason alone. Distinguish: a failure that
  exists without your change is environmental — say so; a render change you made requires
  `./scripts/goldens_docker.sh` (and `--regenerate` only if the change is intended, which is
  Samuel's call). Never use `-DARENAI_REGENERATE_GOLDEN_IMAGES=ON` to get green.

## 3. Done criterion

Every hook `Passed` (or `Skipped` for no matching file) in one full run. Otherwise the work is
not done: quote the failing hook's output and say what remains.
