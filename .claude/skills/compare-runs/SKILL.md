---
name: compare-runs
description: Compare several ArenAI training runs (outputs/train_NNN_*) from their CSV analyses, accounting for code differences (effective commits, diff excluding comments) and config.json differences, then write outputs/comparisons/<date>_<runs>.md. Use for « compare 502 503 504 », « 504 vs 502 », « est-ce que le changement X a servi », « lequel est le meilleur ».
---

# Run comparison

Input: two or more runs (folders or numbers), **reference first**, optionally the question asked
(« le SoftClamp a-t-il servi ? »).
Output: `outputs/comparisons/<YYYY-MM-DD>_<n1>_vs_<n2>[_vs_…].md` (French) + a chat summary.

Role: establish facts, expose options; Samuel decides. Vocabulary (confound, noise floor,
structural effect): `docs/rl_method.md`.

## 1. One up-to-date analysis per run

For each run whose `<run>/analysis.md` is missing or older than the last row of `metrics.csv`
(the comparison script reports it), apply `analyze-run` first. Then read every `analysis.md`:
they hold the confirmed effective commit, what each run tests, and known traps.

## 2. Comparative tables

From the repo root:

```shell
python3 .claude/skills/compare-runs/scripts/compare_runs.py <ref> <run2> [<run3> …] [--step 100000] [--half-width 15000] [--columns hit,kill,ev,Hc]
```

Reuses `analyze-run/scripts/arenai_run.py`. Prints: state and throughput of each run, estimated
effective commit, differing `config.json` keys, code diff excluding comments between consecutive
runs (in the given order), missing CSV columns, D milestones, means **at equal D**, means at
equal index, peaks, signals, state of each `analysis.md`.

## 3. Inventory every difference

A comparison is worth exactly what is known about what differs. For each pair:

- **config.json**: every differing key (nb_tanks, rollout, epochs, lr, entropy targets…);
- **code**: read the real diff between effective commits (`git diff <a> <b> -- <file>`);
  reward, economy, σ/entropy, trainer, curriculum first;
- **outside git**: uncommitted code at launch, different binary, relaunched folder (memory notes
  and `analysis.md`);
- **metrics**: a column whose definition changed between the commits
  (`git log -S'"<name>"' <a>..<b>`) is not comparable as is — say it.

More than one difference ⟹ **confounded**: attribute the gap to none of them. If Samuel needs
the attribution, the isolating control run is a tactical training (gate G4).

## 4. Compare honestly

- **Equal D first** (same curriculum stage), equal index second; tank-steps or wall-clock when
  `nb_tanks` differs. Never beyond the D reached by the least advanced run; flag running runs.
- **Noise floor** (n = 1 seed per config): D milestone ±30k, emptied reserve ×2, hit/shell ±1
  point between identical configs. Require a gap held over several D levels or several saves. An
  identical D calendar is no proof of equivalence (seeded curriculum).
- **Structural vs noise**: an effect mechanically imposed by the code (σ never at the cap with a
  soft bound, α rising when Hc drops under target…) is established; a hit/kill level gap is
  established only beyond the noise floor.
- `ev` in-sample (inflated by more critic epochs), `fire` blind to the dump, `Hd` ≈ H(zoom): do
  not compare these columns naively across configs that change those mechanisms.
- Read `MEMORY.md` and the run notes: no refused lever re-proposed.
- If the CSV cannot decide and a probe can, run it (`probe-checkpoint` on the matching saves of
  each run, same side, same mode) — no permission needed.

## 5. Write the comparison

`outputs/comparisons/<YYYY-MM-DD>_<n1>_vs_<n2>….md`, French:

1. **Runs**: state, index/D reached, effective commit, throughput.
2. **Differences**: config, code (summary of the real diff), outside git, metrics; confounds.
3. **Table at equal D** of the metrics that answer the question (not every column).
4. **Trajectories**: D milestones, peaks, drifts (where and at which D each run breaks away).
5. **Verdict** per question: established / within noise / undecided, with numbers.
6. **To decide**: what it would take (D to reach, probe, control run isolating one variable) —
   options, not decisions.

Then a short chat summary linking the file.
