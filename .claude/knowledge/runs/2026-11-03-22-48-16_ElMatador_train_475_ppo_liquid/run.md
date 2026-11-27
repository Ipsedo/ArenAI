# train_475_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-03 22:48:16 |
| commit at launch | `12e06985` — more shells in reserve + more shells added at hit + no more out of sells death |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_475_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-04: launched 2026-11-03 22:48, 957 k steps, 9 saves: D stayed 250 from start to end. Config = 474 + γ 0.995 + economy driven by progress (30/20/60 → 8/5/16) + ADR thresholds `curriculum_ratio_low/high` from 0.03/0.06 to 0.07/0.10.
- 2026-11-04 cause: the hits/fires ratio probed by `ProgressCurriculum` (24 probe episodes, pooled) stayed in the dead band [0.07; 0.10] all run. CSV `hit`: max 0.0998, 0 lines > 0.10, 44 lines < 0.07 of 1065. Fifth-by-fifth means 0.074 → 0.092. So `upper` never moved. 474 with the same 8 % but high threshold 0.06 rose to 300 then 350.
- 2026-11-04 chicken-and-egg: at progress 0 the economy is the generous one (30/20/60, = 474) where dump is the optimum (see ammo-dump-optimum, run 474). The economy hardening that should make restraint rational only comes with progress, and progress requires hit > 10 % that the « always fire » policy does not give (ceiling 8-9 %; see scarcity-economy-tied-to-competence). The 0.10 threshold had been chosen above that ceiling on purpose; result: nothing moves.
- 2026-11-04 secondary: Hc 0.2 → −0.44, Hd 0.47 → 0.31, kl 0.008 → 0.03 (target 0.05), skip 0 → 21 % in the last quarter, clip 0.11 → 0.25. ev stable 0.4-0.5. Kill/episode 6 → 10, s ≈ 43 s. Not yet the KL regime of 416/449.
- 2026-11-04 **Decision (Samuel):** `--curriculum_ratio_high` 0.10 → 0.08 (default in main.cpp), `ratio_low` 0.07 → 0.05 ⟹ dead band [0.05; 0.08]. Run 476 to compare to 474 (thresholds 0.03/0.06, D 350 at 305 k). Rule: an ADR gate must be reachable by the current policy at progress 0.
