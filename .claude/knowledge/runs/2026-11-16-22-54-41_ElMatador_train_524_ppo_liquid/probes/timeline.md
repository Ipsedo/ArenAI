# Probe chronology: run 524

## Run and code

Run 524 (ppo_liquid, running, 625 k at probe time): flat CNN + single advantage + `target_kl` 0.10 (see analysis.md). Effective code 5c338159 + target_kl 0.10 = f4702f90. Probe binary `build/arenai_agent/arenai_agent_probe`, compiled 2026-11-16 23:07 at f4702f90 = exactly the run's code. The script's "inference diff" warnings come from its estimate of the run's commit (68c41e20, postdating the training binary): probes run with `--force`, no reservation on validity.
Probes: saves 5, 9, 16, 22 (135 k / 243 k / 432 k / 594 k; D 400 / 600 / 950 / 1000), sides 550 and 1000, 4 episodes, stochastic. References at same saves: 522 (s5, s9, s16, s24) and 520 (s9, s16, s20, s24), same `save_every`.

## Chronological table

Side 550 (touches/tank-s = shots/s × hit/shell):

| save | D | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | %empty | %disarmed | tour<5° | σ canon | \|μ\| cx | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s5 | 400 | 4.2 | 0.74 | 0.177 | 11.4 % | 24 % | 24 % | 1.10 | 0.125 | 95 % | 36 % | 15.7 % | 0.46 | 0.47 | 0.46 |
| s9 | 600 | 25 | 0.31 | 0.012 | 32.4 % | 59 % | 28 % | 0.21 | 0.068 | 36 % | 2 % | 12.4 % | 0.60 | 0.53 | 0.69 |
| s16 | 950 | 31 | 0.37 | 0.012 | 28.0 % | 45 % | 30 % | 0.26 | 0.072 | 48 % | 4 % | 10.1 % | 0.51 | 0.43 | 0.66 |
| s22 | 1000 | 13 | 0.35 | 0.027 | 17.7 % | 31 % | 19 % | 0.36 | 0.064 | 84 % | 16 % | 8.3 % | 0.43 | 0.52 | 0.37 |
| *522 s9* | 650 | 2.5 | 0.09 | 0.036 | 21.8 % | 35 % | 45 % | 0.45 | 0.098 | 71 % | 12 % | 24.8 % | 0.26 | 0.20 | 0.77 |
| *522 s16* | 1000 | 7.2 | 0.04 | 0.006 | 46.2 % | 56 % | 71 % | 0.16 | 0.075 | 20 % | 0.1 % | 45.1 % | 0.25 | 0.18 | 0.60 |
| *522 s24* | 1000 | 15 | 0.17 | 0.011 | 36.4 % | 44 % | 69 % | 0.42 | 0.154 | 43 % | 1 % | 38.7 % | 0.35 | 0.22 | 0.58 |
| *520 s16* | 1000 | 8.4 | 0.13 | 0.015 | 30.2 % | 44 % | 55 % | 0.29 | 0.088 | 38 % | 1 % | 33.3 % | 0.30 | 0.17 | 0.11 |
| *520 s20* | 1000 | 13 | 0.29 | 0.023 | 27.6 % | 44 % | 49 % | 0.54 | 0.149 | 41 % | 1 % | 28.9 % | 0.32 | 0.23 | 0.07 |

Side 1000:

| save | D | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | %empty | %disarmed | tour<5° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s5 | 400 | 6.0 | 0.77 | 0.129 | 7.4 % | 20 % | 20 % | 0.78 | 0.058 | 95 % | 39 % | 13.5 % |
| s9 | 600 | 54 | 0.24 | 0.004 | 22.3 % | 43 % | 22 % | 0.07 | 0.016 | 22 % | 1 % | 14.8 % |
| s16 | 950 | 67 | 0.41 | 0.006 | 22.7 % | 47 % | 23 % | 0.09 | 0.020 | 31 % | 3 % | 9.1 % |
| s22 | 1000 | 18 | 0.32 | 0.018 | 10.1 % | 26 % | 13 % | 0.22 | 0.022 | 88 % | 11 % | 6.0 % |
| *522 s16* | 1000 | 14 | 0.03 | 0.002 | 44.1 % | 55 % | 74 % | 0.08 | 0.035 | 13 % | 0.1 % | 36.9 % |
| *522 s24* | 1000 | 36 | 0.18 | 0.005 | 36.1 % | 47 % | 70 % | 0.19 | 0.069 | 34 % | 1 % | 23.2 % |
| *520 s20* | 1000 | 16 | 0.26 | 0.017 | 24.5 % | 41 % | 46 % | 0.32 | 0.079 | 50 % | 1 % | 28.0 % |

CSV at same index: kill 20.2 / 18.9 / 9.6 / 8.6; hit 0.14 / 0.23 / 0.26 / 0.19; skip 0.32 / 0.22 / 0.24 / 0.26; α_fire 1e-4 / 1e-4 / 1.4e-2 / 2.4e-2.

## Reading per axis

- Trigger — ahead of all previous runs, then relaxed. At s9 ratio 25/54 (522 same index: 2.5/2.3; 520: 1.4/1.6) with high on-axis level (p_al 0.31-0.41 from s9 to s22; 522 s16: 0.04, 520 s16: 0.13) and off-axis at floor (0.004-0.012). 520/522's trigger trough (axis extinguishing before recovering) does not exist here. Selectivity carried by distance more than angle: angle-only ratio 2.6-7, identical p_fire at 0-2° and 2-5°. From s16 to s22 off-axis ×2.3 (550) and ×3 (1000), axis unchanged: ratio 31 → 13 and 67 → 18.
- Aim — does not learn, regresses. tour<5° 15.7 → 12.4 → 10.1 → 8.3 % side 550, 13.5 → 14.8 → 9.1 → 6.0 % side 1000. 522 went 13 → 45 % over the same interval, 520 was at 29-33 %. Starting value (13-16 %) is that of an untrained policy (523 s2: 13 %). σ canon stays 0.43-0.60 (522/520: 0.25-0.35) and |μ| canon_x 0.43-0.54 (0.17-0.23): wide noisy turret commands, no saturation (μsat ≤ 15 %).
- Shot — precise when it leaves, rare because opportunity is rare. Aligned shots hit 59 % at s9 (522: 35 %), but only 19-30 % of shots are aligned (522: 45-74 %, 520: 46-55 %). hit/shell 32 → 28 → 18 % (550), 22 → 23 → 10 % (1000). Touch throughput flat and low: 0.064-0.072 touches/tank-s side 550 from s9 to s22 (522 s24: 0.154; 520 s20: 0.149), 0.016-0.022 side 1000 (0.069 / 0.079).
- Dump — back at s22. %empty 36 → 48 → 84 % (550), 22 → 31 → 88 % (1000); disarmed 2 → 4 → 16 % and 1 → 3 → 11 %. 522 and 520 stayed ≤ 1-3 % disarmed at D 1000.
- Heads. p_zoom 0.69 → 0.37 between s9 and s22. σ never at ceiling.
Order of events: trigger becomes selective (before 243 k); aim never takes off and erodes monotonically; α_fire takes off (340 k); off-axis rises and dump returns (432 → 594 k).

## Link with CSV and code

- The CSV kill drop (23 → 8 between 170 k and 340 k) occurs with a selective trigger and high precision per aligned shot: it comes from the number of opportunities, hence aim. The curriculum moves targets away (D 450 → 800); a turret that does not search crosses fewer near enemies on axis. Decisions at < 5° go from 18.4 k to 10.6 k side 550 between s9 and s22 for a rising number of decisions.
- CSV hit drop at D 1000 (0.255 → 0.18) and doubled emptyings = the relaxation seen at s22, synchronous with α_fire ×290. Mechanism of the "trigger entropy regulator" note, here with hold because gain per shot is low.
- Code: in 520/522 canon actions received the λ-return of the aim potential (separate credit, group KL stop). 5c338159 puts them back on the critic's GAE advantage, with a single KL stop cutting 20-33 % of minibatches. Aim is the only axis that lost its dedicated signal and the only one regressing; the trigger, credited by the same advantage, does better than before.

## Established / noise / not settled

Established (monotone trend over 4 saves, 2 sides, ×3-4 gap to references): turret does not aim and regresses, it caps the run; trigger selective earlier and at higher level than in 520/522; relaxation between s16 and s22 (off-axis ×2-3, dump back), carried by α_fire. Noise: exact side-1000 ratio values (aligned & close n = 319-435). Not settled: share of flat CNN, single advantage and single KL stop in aim loss (the other PC's run, flat CNN + separate credit, is the control); whether the early trigger comes from single advantage or CNN (523 s2 decoded left/right at 0.75-0.79); deterministic mode not probed (from p > 0.5, 4-7 % of aligned states fire side 550).
Next (options): probe same save on the other PC's run: tour<5° ≥ 25 % at D 600-1000 there would designate single advantage, not the CNN; measure ρ(advantage, corrective gesture) with `sigma_gradients` on s9 or s16 (threshold 0.08); decode yaw (`--features`) at s16; continuing 524 adds no new information while tour<5° falls; marker of reversal: tour<5° > 15 % side 550 at a later save.
