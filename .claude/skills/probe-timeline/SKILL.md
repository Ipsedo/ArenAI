---
name: probe-timeline
description: Analyse the chronology of the probes of one ArenAI training (<run>/probes/*.json, successive saves), crossed with the CSV analysis, the config and the run's code, then write <kdir>/probes/timeline.md. Use for « analyse les sondes de 504 », « comment évolue la gâchette au fil des saves », « la sélectivité tient-elle de save_13 à save_27 ».
---

# Probe timeline of a run

Input: a run (folder or number), optionally a subset of saves.
Output: `<kdir>/probes/timeline.md` (`<kdir>` = the run folder of the knowledge base, CLAUDE.md) + a chat summary.

Role: establish facts, expose options; Samuel decides. Vocabulary: `docs/rl_method.md`.

## 1. Prerequisites

- **Probes**: results of `probe-checkpoint` (`<run>/probes/save_K_side<S>.json`). Fewer than two
  saves probed on the same side ⟹ no chronology. Choose the saves that discriminate (at D
  changes, around a CSV peak or break, plus `save_0` as random reference) and **probe them
  yourself** — fast evaluators need no permission; state the GPU cost.
- **CSV analysis**: `<kdir>/analysis.md` from `analyze-run`. Missing or older than the last row
  of `metrics.csv` ⟹ apply `analyze-run` first. It provides the effective commit, what the run
  tests, the phases and regimes of the CSV.

## 2. Tables

```shell
python3 .claude/skills/probe-timeline/scripts/probe_timeline.py <run> [--saves 5,9,13] [--half-width 15000]
```

Prints:
- the binaries used, with a warning when the chronology mixes binaries or contains a probe made
  despite an inference diff;
- one table per side and per mode (stochastic or deterministic, episode count), one row per
  save, with the `probe-checkpoint` metrics and their counts;
- the CSV metrics around each save's index: D, hit, kill, fire, ev, Hc, Hd, kl, skip and the
  highest α;
- the saves not probed.

## 3. Analyse

1. **Comparability.** On one curve, only probes with the same side, same mode, same episode
   count and equivalent inference code. If the binary changed mid-chronology, read its diff
   (`probe-checkpoint` §1) before comparing saves across the change.
2. **Trajectory per axis**, kept separate:
   - trigger: ratio, p_al, p_off, level, deterministic mode;
   - shooting: hit/shell, share of aligned shots, enrichment;
   - dump: emptied reserve, % emptying, % disarmed;
   - aim: turret < 5°;
   - heads: σ, |μ|, saturation, zoom.

   State which axis moves **first** (e.g. the turret lets go before the trigger; μ saturation
   precedes the loss of aim): order of events is the causal evidence.
3. **Cross with the CSV and the code.**
   - Does each probe change have a counterpart in the CSV (hit, kill, ev, Hc, α, skip) at the
     same index? Or is the CSV blind there (`fire` does not see the dump, `Hd` ≈ zoom)?
   - A change of D between two saves changes the difficulty: a drop on side 550 is no
     regression if D doubled. Read both sides.
   - Tie every mechanism to the run's code: reward, economy, σ parametrisation, α, what the run
     tests per `analysis.md`. A code-imposed effect is structural; the rest is judged against
     the noise floor.
4. **Noise.** Ratio unreliable under ~300 aligned-and-close decisions. 4 episodes per probe,
   n = 1 seed: require a trend over ≥ 3 saves, or a gap clearly beyond the variability between
   neighbouring saves. Markers of the run notes (records, other runs) compare at the same D and side.
5. **Prior knowledge.** Read `.claude/knowledge/INDEX.md` and the notes of the run and its predecessors. No refused lever
   re-proposed.
6. Any cause or remedy drawn from the timeline follows `rl-hypothesis`.

## 4. Write `<kdir>/probes/timeline.md`

English:

1. **Run and code**: effective commit, what the run tests, probe binary(ies).
2. **Chronological table** of key metrics per side, with D and the CSV at the same index.
3. **Reading per axis**, with the order of events.
4. **Link with the CSV and the code**: mechanisms, with their evidence.
5. **Established / within noise / undecided.**
6. **Next**: saves to probe to decide, markers to watch — options, not decisions.

Then a short chat summary. A newly established fact updates `<kdir>/run.md`.
