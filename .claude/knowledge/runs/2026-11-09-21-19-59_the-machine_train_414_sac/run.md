# train_414_sac

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-09 21:19:59 |
| commit at launch | `2e594069` — clamp alpha to 0.001 |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_414_sac` on the-machine |

## Established facts

_SAC run 414 (2026-11-09/10) — first SAC above the random floor; lock lifted = α decay down to ~5e-4 (45k steps lost); hit peak 0.117 at D=300 (save_3) then regression to 0.067 with Q overestimated +0.4_

Run 414 (config: networks 128/64, batch 512, gpu_batch 1024, gradient_steps 2, α₀ 0.01,
lr α 3e-5, target_sigma 0.1, fire/zoom init proba 0.5, 32 tanks, 162k steps in 10h40,
27k steps ≈ 1h47). Saves every 27k: save_3 = 81k = best.

Three phases:
1. 0–45k: Hc 2.7→2.1 (σ≈0.9), hit 0.04–0.05 = random floor, kill 1→0.1. α decays
   at Adam's max speed (3e-5/update in log): 0.01 → 6e-4. **Nothing learns while
   α > ~5e-4**; 412 (gradient_steps 1) had only reached 2.6e-3 at 45k → same floor.
2. 45k–80k: as soon as α ≈ 5e-4, the Q term dominates, Hc collapses 2.1 → −3.5 (σ target 0.1 reached
   at ~78k, target Hc −3.53), hit 0.05 → 0.117, kill → 5, curriculum 250 → 300 (68k).
3. 80k–162k: α_c ≈ 7e-5 (at target, no effective regularisation), hit 0.117 → 0.094 at
   constant D=300, then 350 (115k) → 0.067; kill 6.8 → 2.7; miss 25 → 28 m; fire rises
   0.5 → 1.0 (118–133k); critic loss q ×6 (0.004 → 0.026); mean Q 0.33–0.43 while
   realised return per life ≈ 0 (zero-sum reward, r ≈ −0.0003/step) → **overestimation ≈ +0.4**.

**Why:** the dead-phase duration is purely α's decay time (lr α × number of
updates), not an exploration problem; the later regression resembles the classic
SAC drift without entropy (near-deterministic policy exploiting critic errors).

**How to apply:** options to validate with Samuel, not decided: α₀ ≈ 5e-4 or lr α ×10 to
skip the dead phase; separate regression and curriculum (D fixed at 300); regularise the critic
(target_sigma 0.2, gradient_steps 4, Q clip); test save_3 in game. See
[sac-train-412-diagnosis](../undated/sac-train-412-diagnosis.md), sac-agent-2026-11.
