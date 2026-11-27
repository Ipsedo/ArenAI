# Game-verdict benchmarks: Samuel's verdicts + game-rules matches (`arenai_agent_match`) per measured checkpoint

Runs: 546, 550, 551, 552, 553, 554, 555

Samuel's verdicts after playing a checkpoint (game labels, `docs/rl_method.md` M7); game-rules match reading of every measured checkpoint; current reference = 552 s41. (Samuel's verdicts are tracked in `.claude/knowledge/game_labels.md`.)

Samuel's verdicts in game:

| checkpoint | date | verdict | his words |
|---|---|---|---|
| 553 save_35/36 | 2026-11-25 | bad | « run pourri en jeu »; reserve empty very fast, canon pointing down |

On 2026-11-25 Samuel has no recent checkpoint he judges surely good (546 s20 delivered not retained as benchmark).

Game-rules matches (8 matches, 16 tanks, seeds 0-7, side 1000 / 500), mean over matches:

| checkpoint | dry at 10 s | dry at 30 s | hits/tank | kills/tank | player killed |
|---|---|---|---|---|---|
| 552 s41 | 0.06 / 0.08 | 0.16 / 0.33 | 4.8 / 9.3 | 0.27 / 0.57 | 0.75 / 0.88 |
| 553 s36 | 0.30 / 0.48 | 0.40 / 0.72 | 5.1 / 8.3 | 0.28 / 0.47 | 0.75 / 0.38 |

Series 2026-11-25 (`outputs/comparisons/2026-11-25_match_546_552_553.md`), dry at 10 s · hits/tank · armed canon elevation, side 1000 / 500: 546 s20 0.14/0.22 · 2.4/5.4 · +1.8/−1.9°; 552 s10 0.02/0.09 · 3.0/5.2 · −21°; 552 s26 0.00/0.00 · 3.2/9.4 · −14/−11° (hit/shell 0.63); 553 s8 0.46/0.60 · 2.7/5.6 · −35°; 553 s20 0.03/0.14 · 3.5/6.8 · −21/−9°. ⟹ 553 empties at s8 and s36 but not at s20; canon low in game on early/mid saves of both runs; 546 s20 (delivered) below 552 s41; no good checkpoint (47-77 % of tanks without hit in 5 min at side 1000). Binary 546: `outputs/train_546_ppo_liquid/match_bin` (`1720af12`).

**Why:** the instrument ranks the 553 benchmark in Samuel's direction (reserve emptied); no positive benchmark ⟹ not validated on the « good » side.

**Rule:** current reference for any new checkpoint = 552 s41 (`outputs/train_552_ppo_liquid/match_bin`, `LD_LIBRARY_PATH` on this folder). Add a line for every Samuel verdict and every measured checkpoint. See runs 553 (spatial aim cell + offset), 552 (presence head, canon low).

Series 550/551/552 of 2026-11-25 (`outputs/comparisons/2026-11-25_match_550_551_552.md`), 550 = twin of 552 without auxiliary head: 550 s28 (774 k) dry at 10 s 0.02/0.01 · hits 5.2/8.1 · kills 0.23/0.44 · armed canon −0.4°; 550 s26 5.5/8.8; 550 s10 2.95/5.2 at −2.4°. 551 s35 3.5/7.8. ⟹ the early/mid low canon is specific to 552 (head), absent in 550/551; end of run 552 s41 ≈ 550 s28 in combat, 550 s28 stays armed longer. Binaries: `outputs/train_550_ppo_liquid/match_bin` (`65899e8a`), `outputs/train_551_ppo_liquid/match_bin` (same binary + `libarenai_agent.so` from `probe_bin`, direct branch).

554 (panorama), 2026-11-26, dry at 10 s · hits/tank · kills/tank, side 1000 / 500: s3 0.50/0.56 · 0.87/2.39 · 0.07/0.19; s10 0.14/0.20 · 2.10/4.97 · 0.17/0.37; s17 0.04/0.05 · 5.07/7.62 · 0.29/0.46 (≡ 550 s26, below 552 s41). See run 554.

555 (panorama mean), 2026-11-26, dry at 10 s · hits/tank · kills/tank, side 1000 / 500: s15 0.14/0.18 · 1.53/3.95 · 0.08/0.20; reference at equal index 554 s15 0.05/0.07 · 4.14/6.62 · 0.26/0.45. s17 0.03/0.02 · 5.39/9.45 · 0.27/0.49 (≡ 554 s17, level of 552 s41). See run 555.
