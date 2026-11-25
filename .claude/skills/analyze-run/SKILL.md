---
name: analyze-run
description: Analyse the metrics.csv of an ArenAI training folder (outputs/train_NNN_*) against the exact code of the run (effective commit, diff vs the previous run) and its config.json, then write <run>/analysis.md. Use whenever a run or its CSV must be read (« analyse 504 », « où en est le run », « regarde le CSV de train_503 », « qu'est-ce qui s'est passé dans ce run »).
---

# Run analysis

Input: `outputs/train_NNN_<agent>/` or its number (`504`).
Output: `<run>/analysis.md` (French) + `<run>/analysis_summary.json`, and a short chat summary.

**Train rules only** (`docs/rl_method.md` M7): nothing read here supports « bon / meilleur en jeu »;
a verdict on the run needs skill `game-match`.

Role: establish facts, expose options. Deciding is Samuel's (`CLAUDE.md` gates). Reading
discipline, noise floor and vocabulary: `docs/rl_method.md`.

## 1. Numeric summary

From the repo root:

```shell
python3 .claude/skills/analyze-run/scripts/analyze_run.py <run> --json [--prev <reference_run>]
```

Stdlib only. Prints: dating (mtime of `config.json`, never of the folder — folders get reused),
running/finished, throughput in steps/h and tank-steps/h, commit at launch and commits during
the run, config diff and code diff **excluding comments/whitespace** against the previous run, D
milestones, means per D level, 10 index slices, extremes, automatic signals. Signals are leads,
not verdicts.

## 2. Establish the exact code of the run

1. **Effective commit.** Samuel often launches before committing: the script flags commits ≤ 90
   min after launch (and before the next run) as "may contain the run's code" and derives an
   estimated effective commit. Confirm it:
   - `config.json` values must equal the defaults of
     `arenai_agent/src/agents/<agent>/*_hyperparams.h` at the effective commit
     (`git show <hash>:<path>`), except explicit CLI options;
   - read the run's memory note and its neighbours': they record "uncommitted", "relaunched",
     "binary of HH:MM"… Memory and git disagreeing is a finding: state it;
   - running run + dirty tree (`git status`): compare the binary mtime
     (`cmake-build-release/arenai_agent/arenai_agent_train` or `build/…`) to the launch date.
2. **Read the code at the effective commit, never the working tree** (`git show <hash>:<path>`)
   for everything `config.json` does not hold:
   - reward and shaping: `arenai_agent/src/core/train_environment.cpp`;
   - events, shell economy: `arenai_model/src/tank/jolt_enemy_tank.cpp`,
     `arenai_model/include/arenai_model/constants.h`, `shell_economy.h`;
   - σ, entropy, α: `arenai_agent/src/networks/constants.h`, `networks/misc.cpp`,
     `networks/entropy.cpp`, `agents/entropy_alphas.cpp`;
   - the agent's trainer (`agents/<agent>/*_trainer.cpp`), curriculum
     (`core/spawn_curriculum.cpp`), metric definitions (`MeanMetric("<name>", window, …)`
     constructors).
3. **What differs from the previous run**: read the real diff of the files the script lists
   (`git diff <from> <to> -- <file>`), not commit messages. Name the change(s) under test.
   Several ⟹ **confounded**: say it, attribute nothing.

## 3. Read the CSV

One row every `30 / wanted_frequency` iterations; `index` = training-loop iterations = decisions
(period `action_repeat × wanted_frequency`, 0.1 s by default). Row `index = 0` is ignored.
`save_N` ≈ `N × save_every`. Columns change between runs: trust the actual header and, for an
unknown or suspicious column, its definition at the effective commit.

| column | meaning (as of 2026-11-11 — re-verify at the run's commit) |
|---|---|
| `r`, `r_hit`, `r_rcv`, `r_aim` | mean reward per tank-decision, very long window (4 × nb_tanks × max_episode_steps) ⟹ smoothed, lagging |
| `miss` | mean miss distance (m) of landed shells |
| `s` | lifetime (s) of dead tanks |
| `fire` | shots/s per living tank, 256-decision window; **blind to the dump** at episode start |
| `hit` | hits/shots per episode, 16-episode window (≠ the probe's hit/shell) |
| `kill` | kills per episode, 16-episode window |
| `side`, `D` | spawn side played / upper bound of the curriculum |
| `π`, `v` | actor / critic losses |
| `ev` | explained variance **in-sample** (all minibatches of all epochs): more critic epochs inflate it |
| `Hc` | mean continuous entropy over dims (hides one saturated dim) |
| `Hd` | discrete entropy; fire masked outside armed states ⟹ ≈ H(zoom) |
| `clip`, `kl`, `skip` | clipped fraction, KL per rollout, fraction of minibatches skipped by `target_kl` |
| `alpha_*` | entropy temperature per action dimension (CSV only) |

## 4. Interpret

- Read `MEMORY.md` and the linked notes (lineage runs, feedback). **Never** re-propose a lever
  recorded as refused, nor a cause recorded as ruled out.
- Compare at **equal D**, not equal index. Noise floor: D milestone ±30k, reserve ×2; one
  isolated record proves nothing. Only code-imposed effects are structural.
- Throughput at different `nb_tanks`: compare in tank-steps or wall-clock.
- Regimes to recognise: KL regime (skip ↑, kl ≥ target), σ collapse (Hc ↓ < 0), weak critic
  (ev < 0.2; ev leads hit by 10–20k), trigger off (fire → 0, kill → 0), dump (invisible in the
  CSV: reward decomposition or probe), μ at the bounds, α off its floor.
- The CSV says nothing about trigger selectivity or the emptied reserve. If the question needs
  it, **run the probe** (`probe-checkpoint`; fast evaluators need no permission) instead of
  concluding or merely suggesting.
- Every quantitative statement carries its index and D.
- Any cause or remedy that comes out of the analysis follows `rl-hypothesis` (fast evaluator
  attached, gates applied).

## 5. Write `<run>/analysis.md`

French, no filler, LaTeX for any formula:

1. **Header**: run, agent, launch → last row, running/finished, final index, throughput,
   effective commit (with confidence).
2. **What this run tests**: config + code diff vs the previous run, hypothesis (memory),
   confounded changes, any typed imposition by Samuel.
3. **CSV reading**: D milestones, trajectory by phase, key numbers at equal D against the
   reference runs known in memory.
4. **Diagnosis**: mechanisms with their evidence; separate established / within noise /
   undecided.
5. **Options** (no decision), each with its fast evaluator, and **markers**: what to watch, at
   which D/index, which probe.

Then a short chat summary linking the file. A newly established fact updates (or creates) the
run's memory note.
