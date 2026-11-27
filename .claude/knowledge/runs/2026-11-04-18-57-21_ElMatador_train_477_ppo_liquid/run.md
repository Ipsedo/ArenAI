# train_477_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-04 18:57:21 |
| commit at launch | `324d21a9` — curriculum on shells economy |
| commits ≤ 90 min after | `5c4d1114` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_477_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-04: centred shot reward (run 477): baseline b = EMA of (aim + hit) per shell subtracted at impact (see centred-shot-reward note). Probe save_1 (108 k): trigger neither dead nor dump (p_fire 0.06, 0 % > 0.5 ⟹ mute in deterministic) but flat vs angle; turret aligned 2.3× more than init (< 5°: 11.7 % vs 5.1 %); hit per shell 12 % vs 6 %; reserve emptied for 52 % vs 84 % of tanks. Details: `probes/save_1.md`.
