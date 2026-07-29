---
name: final-check
description: Final verification of code modifications via `pre-commit run --all-files` — formats the code (clang-format, whitespace), builds the project and runs all module tests. ALWAYS use this as the last step after modifying C++ code, before declaring the work done or committing (e.g. "verify", "final check", "run pre-commit", "make sure everything passes").
---

# Final check (pre-commit)

Run the full pre-commit pipeline on the whole repo. The hooks (defined in the parent `ArenAI/.pre-commit-config.yaml`) do, in order: `end-of-file-fixer`, `trailing-whitespace`, `clang-format`, then **build** (`build_linux.sh`) and the **test binaries** of each module (view, model, core, agent). `fail_fast: true` — the run stops at the first failing hook.

## Prerequisites

`pre-commit` lives in the venv at `arenai/.venv`. If it's missing, run the `setup-venv` skill first (from `arenai/`).

## Steps

1. Run from the **git root** (`ArenAI/`, the directory containing `.pre-commit-config.yaml`), using the venv binary directly:
   ```shell
   ./.venv/bin/pre-commit run --all-files
   ```

2. **Interpret failures** — a hook marked `Failed` means one of two things:
   - **Formatting hooks** (`end-of-file-fixer`, `trailing-whitespace`, `clang-format`): they *fixed* files in place and report `Failed` because files were modified. This is normal — re-run the command; it should pass on the second pass. Review the reformatted files (`git diff`) if the changes touch code you just wrote.
   - **`build` or `test_*` hooks**: a real compilation or test failure. Read the error output, fix the code, then re-run from step 1.

3. **Repeat until everything passes.** The task is only done when a full run shows every hook `Passed` (or `Skipped` for hooks with no matching files).

## Notes

- The build hook is incremental (reuses `./build/`), so re-runs are much faster than the first one; still allow several minutes for a cold build. Use a generous Bash timeout (≥ 10 min).
- Report the outcome honestly: quote the failing hook's output if something fails; only claim success after a fully green run.
- Formatting fixes made by the hooks are part of your change — don't revert them, and include them in any commit.
