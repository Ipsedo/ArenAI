# Run 543 — constant empty-reserve penalty 0.5 (point at 176 k)

## Header

`ppo_liquid`, launched 2026-11-20 18:54, last line 20:57, **running**, index 176 k, D 450. Throughput 86 k steps/h (2.75 M tank-steps/h). Code: uncommitted working tree above `b83da585`. Binary `cmake-build-release/arenai_agent/arenai_agent_train` dated 18:54, started at 18:54:17, and `train_environment.cpp` carries `EMPTY_RESERVE_PENALTY = 0.5f`: good confidence, not provable by git.

## What the run tests

541 + constant empty-reserve penalty 0.5 (before: D/1000, i.e. 0.25 at D 250 and 0.8 at D 800). `config.json` identical to 541 (`discrete_target_proba` 0.01). One gap.
Mechanical consequence: penalty ×2 at D 250, ×1.1 at D 450, equal at D 500, then weaker than 541 (×0.63 at D 800). The zone the change targets (trigger holding at D 750-950) is not reached.

## CSV

D milestones identical to 541, 542 and 539 to the thousand (300 @ 47 k, 350 @ 67 k, 400 @ 114 k, 450 @ 144 k): the kills/tank criterion passes at every probe, the curriculum advances at maximum cadence and separates no run before D ~700.
Means per D level (543 / 541 / 542 / 539):

| D | kill | hit | fire (shots/s) | r_hit ×1e-3 |
|---|---|---|---|---|
| 250 | 12.5 / 14.8 / 11.2 / 14.5 | 0.082 / 0.088 / 0.074 / 0.085 | 1.02 / 1.07 / 1.02 / 1.41 | 3.7 / 4.5 / 3.3 / 5.3 |
| 300 | 21.8 / 19.7 / 23.0 / 22.4 | 0.118 / 0.103 / 0.114 / 0.120 | 1.03 / 0.90 / 1.93 / 1.65 | 5.8 / 5.4 / 6.8 / 8.1 |
| 350 | 20.4 / 20.5 / 24.7 / 24.8 | 0.111 / 0.101 / 0.117 / 0.122 | 0.84 / 1.00 / 1.56 / 1.85 | 5.4 / 4.9 / 8.8 / 9.8 |
| 400 | 20.0 / 19.7 / 25.2 / 26.2 | 0.105 / 0.106 / 0.120 / 0.118 | 0.74 / 0.89 / 1.68 / 2.04 | 4.3 / 4.8 / 9.6 / 10.9 |
| 450 | 19.7 / 22.3 / 24.9 / 25.8 | 0.095 / 0.119 / 0.116 / 0.113 | 1.04 / 1.66 / 1.35 / 1.85 | 3.7 / 6.1 / 8.9 / 10.8 |

Empty penalties (`rest` = r − r_hit − r_rcv) and interval between two emptyings per tank (penalty / |rest| × 0.1 s):

| D | rest 543 | rest 541 | rest / r_hit 543 | emptying 543 | emptying 541 |
|---|---|---|---|---|---|
| 250 | −3.4e-3 | −1.9e-3 | 0.92 | 14.7 s | 12.9 s |
| 300 | −3.6e-3 | −2.0e-3 | 0.62 | 13.9 s | 15.3 s |
| 350 | −3.3e-3 | −2.3e-3 | 0.62 | 15.0 s | 15.4 s |
| 400 | −2.9e-3 | −2.4e-3 | 0.67 | 17.5 s | 16.7 s |
| 450 | −2.9e-3 | −3.0e-3 | 0.79 | 17.0 s | 15.3 s |

`r` oscillates around 0 (max +4.6e-4 at 64 k, −4.4e-4 mean at D 450); 541 stayed at +1e-3. kl 0.014 → 0.028, skip 0, skip_canon < 0.005, ev 0.40-0.50, Hd 0.69 → 0.66.

## Diagnosis

Established: the trigger did not extinguish: fire 0.7-1.1 shot/s throughout (min 0.44 at 142 k, last row 0.66), kill 20-23, last point 22.8 (run max). The 479 risk is ruled out at this stage. The emptying frequency did not move: one emptying every 14-17 s per tank, as 541, although the penalty is doubled at D 250. The dump is not resorbed, it only costs more, hence `r` ≈ 0 and a `rest` worth 60-90 % of r_hit (541: 36-50 %).
In seed noise: kill and hit ≡ 541 up to D 400. At D 450, 543 is under 541 (r_hit 3.7 vs 6.1e-3, fire 1.04 vs 1.66): one step, 32 rows.
Not settled: the targeted effect. 543's penalty becomes weaker than 541's only beyond D 500; the collapses of 538-542 came at D 750-850. With 541's milestones (D 700 at 281 k) ~350-450 k are needed, 2 to 3 h more. 543 and 541 are both under 542 and 539 from D 350 to D 450 (kill 20 vs 25). 542 differs from 541 only by the discrete target, so the 541/542 gap gives the order of magnitude of noise between two runs.
Options/markers: let run to D 800-950, where the change reads (fire, kill, `rest` / r_hit vs 541: 39 % at D 800); probe save_6 (D 450) for reserve and trigger selectivity, which the CSV does not show; direct comparison with 541's probes on same side.
