---
name: probe-checkpoint
description: Run the probe (arenai_agent_probe) on an ArenAI checkpoint (outputs/train_NNN_*/save_K) after checking that the binary matches the run's code, then analyse its metrics (trigger vs alignment, hit/shell, emptied reserve, turret, σ/μ) and write <kdir>/probes/save_K.md. The standard fast evaluator of a hypothesis — launch it without asking. Use for « sonde save_9 de 504 », « lance la sonde sur ce checkpoint », « est-ce que la gâchette est sélective à save_12 ».
---

# Checkpoint probe

Input: `outputs/train_NNN_<agent>/save_K` (or a run + `--save K`), optionally spawn sides, number
of episodes, deterministic mode.
Output: `<run>/probes/save_K_side<S>[_det].{csv,_shots.csv,log,json}`, `<kdir>/probes/save_K.md` (`<kdir>` = the run folder of the knowledge base, CLAUDE.md)
(French) and a chat summary.

**Train rules only** (`docs/rl_method.md` M7): nothing read here supports « bon / meilleur en jeu »;
a verdict on the run needs skill `game-match`.

Role: establish facts, expose options. A probe is a **fast evaluator** (`docs/rl_method.md` M2):
launching it, and building the binary it needs, requires no permission.

## 1. Binary ↔ checkpoint compatibility (before any launch)

```shell
python3 .claude/skills/probe-checkpoint/scripts/probe_checkpoint.py <save> --check-only
```

The probe replays the checkpoint **with the code compiled into the binary**, not the run's:
network, distributions, observation and environment (respawn on empty reserve, economy…) come
from the binary. The script picks the most recent binary among `cmake-build-release/` and
`build/` (`--binary` to force one), dates it (mtime → commit) and prints:

- the checkpoint's σ/μ head (`actor_repr.txt`: `Clamp` vs `SoftClamp`, `mu` Linear vs
  Sequential…);
- the diff excluding comments between the run's effective commit and the binary's on the
  inference code, plus uncommitted changes older than the build.

On a diff the script refuses to launch. Read the diff: if it changes the shape or the
interpretation of a tensor (σ parametrisation, `mu` layer, observation size or order), the probe
is **wrong** — and loading may still succeed (`Clamp`/`SoftClamp` did). Then either build a
binary at the right commit (git worktree + skill `build`), or convert a *copy* of the checkpoint
(`arenai_agent/python/convert_mu_linear.py`). If the diff has no effect on inference (refactor,
unchanged environment), relaunch with `--force` and state it in the analysis.

## 2. Launch

```shell
python3 .claude/skills/probe-checkpoint/scripts/probe_checkpoint.py <save> [--sides 550,1000] [--episodes 4] [--deterministic]
```

Defaults: sides 550 and 1000, 4 episodes, stochastic, `--cuda`. The shell economy (`shells`,
`shells_per_hit`, `max_shells`) comes from the run's `config.json`, not the game defaults.
~40 s per episode and per side at 32 tanks (≈ 5–6 min with defaults). Launch with
`run_in_background` and wait for the notification. The probe shares the GPU with any running
training: say so in the report. `--analyze-only` re-analyses existing CSVs without relaunching.

## 3. Metrics (armed decisions = `can_fire` and reserve > 0, step ≥ 10)

One step = one decision = `action_repeat × wanted_frequency` = 0.1 s by default. Durations in
seconds recorded in the run notes before 2026-11-11 are 3× too large.

| metric | meaning |
|---|---|
| `ratio`, `p_al`, `p_off` | armed p_fire aligned (< 5° **and** target < 40 m) / off-axis (≥ 20°): trigger **selectivity**, definition of the records in the run notes since 2026-11-10. `ratio_angle_only` (< 5° alone) is in the JSON |
| `p_arm` | mean trigger level; flat vs angle = no selectivity |
| deterministic (p > 0.5) | what the game does in deterministic mode: a trigger under 0.5 everywhere is mute in game |
| `hit/ob`, `h<5°`, `tirs<5°`, `enrich` | hits per landed shell, per shell fired aligned, share of aligned shots, enrichment (share of aligned shots / share of aligned decisions) |
| `vide_s`, `%vident`, `%désarmé` | first empty reserve (median, s), share of tank-episodes that empty it, share of disarmed steps: the **dump** |
| `tour<5°` | share of decisions with the turret within 5° of a target: aim |
| `σx`, `σy`, `σ@plaf`, `|μ|cx`, `μsat` | median canon σ, share at the cap (≥ 0.69), median canon \|μ\|, share \|μ\| > 1.5 (tanh saturation) |
| `p_zoom`, corr(p_fire, p_zoom) | trigger/zoom coupling |

Check the counts (`n` aligned & close): under ~300 the ratio is noise.

## 4. Interpret

- Place the checkpoint: index = K × `save_every`, curriculum D at that index, CSV around it
  (`analyze-run` if `<kdir>/analysis.md` is missing or stale).
- Compare to the markers of the run notes (`.claude/knowledge/`) **at the same D and the same side**, within the noise floor.
  One isolated record establishes nothing. `save_0` is the random reference.
- Side 550 = mid-range competence, side 1000 = cruise conditions. Read both.
- Aim learning shows as a **negative** correlation between action mean and signed error; measure
  it per error bucket, never globally.
- Known traps: trigger flat vs angle at any level = no learning; raw selectivity distorted by
  the spawn reflex, hence step ≥ 10; "broken turret" in game = usually the empty reserve; CSV
  `Hd` ≈ zoom entropy, so the trigger is readable here only.
- When this probe was run to decide a hypothesis: apply the decision rule written beforehand
  (prediction if true / if false) and state the verdict, refutation included.

## 5. Write `<kdir>/probes/save_K.md`

French. Header (run, save, index, D, binary + commit, compatibility, sides/episodes/mode),
summary table, reading (trigger, shooting, dump, aim, heads), comparison to markers at the same
D, conclusions established vs noise, next probes or saves worth running. Short chat summary. A
newly established fact updates `<kdir>/run.md`.
