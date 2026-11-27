---
name: game-match
description: Judge an ArenAI checkpoint under the desktop game's rules — play matches with arenai_agent_match (no refill of a dry tank, deterministic agent, 16 tanks, idle player tank, game shell economy), read the game outcomes and write <kdir>/matches/save_K.md. The only instrument allowed to support « bon / meilleur en jeu » (docs/rl_method.md M7). Use whenever a run or checkpoint must be judged, ranked or declared improved, before telling Samuel which save to play, and for « est-ce que ce run est bon », « bon en jeu », « lequel est le meilleur », « ça donne quoi en jeu », « évalue le run ».
---

# Game match — the verdict instrument

Read `docs/rl_method.md` M7 first. A game-match result is a **prediction** of Samuel's verdict;
the verdict is his (game labels: `.claude/knowledge/game_labels.md`).

What the executable does (`arenai_agent/probe/match.cpp`): `MatchEnvironment` replays
`DesktopGameEnvironment::play_frame` + `GameSession::play_frame` headless — agent built by the
public `AgentFactory` (deterministic, the game's loading path, fresh per match),
`GAME_SHELL_ECONOMY`, no respawn, idle player tank at the game spawn, decide/observe cadence of
the game. A match ends at `--max_seconds` (300), when no enemy is alive, or when no living enemy
has a shell left (`all_dry`). Known gaps with a real game: the player does not move or shoot;
no line-of-sight test in « trouve »; the match goes on after the player's death.

## 1. Binary ↔ checkpoint

The network and the environment come from the binary. Use `build/arenai_agent/arenai_agent_match`
only for a run whose effective commit equals the binary's on inference and model code
(`probe-checkpoint` §1 check applies). Otherwise build `match.cpp` at the run's commit: git
worktree in the scratchpad, copy `arenai_agent/probe/match.cpp` and its CMake stanza, `make
arenai_agent_match`, copy the executable and every `lib*.so` to `<run>/match_bin/`, remove the
worktree; run with `LD_LIBRARY_PATH=<run>/match_bin`. `match.cpp` uses public API only, it
compiles unchanged back to `db9a825a` (552).

## 2. Play

```shell
<bin>/arenai_agent_match <agent> --run_folder <run> --save save_K --resources_folder resources \
    --output <run>/matches/save_K_side<S> --nb_matches 8 --spawn_side <S> --cuda
```

Sides **1000 and 500**, 8 matches each, seeds 0–7 (default) so two checkpoints face the same
spawns. Measured 2026-11-25 with a training on the same GPU: ~65 s per match, **~9 min per side
of 8 matches, ~18 min per checkpoint**; announce it at launch (`CLAUDE.md` « Answers to Samuel »).
Free (fast evaluator).

## 3. Read

```shell
python3 .claude/skills/game-match/scripts/match_metrics.py <name>=<prefix> [<ref>=<prefix> …] [--json out.json]
```

Mean ± sd over matches, per criterion, in Samuel's order (2026-11-25):

| criterion | metrics |
|---|---|
| trouve les ennemis | armed time with the canon axis < 5° of a living target; time to first target < 5° |
| reste dangereux longtemps | share of living tanks with an empty reserve at 10 / 30 / 60 / 120 s; living time spent dry; armed tanks at the end; matches ending `all_dry` |
| touche le joueur | impacts on the idle player, first impact, player killed, time of death |
| touche et tue les autres | hits / tank, kills / tank, shells / tank, hit / shell, tanks without a hit |
| tenue visible | canon elevation armed and dry, armed share with canon < −5° |

## 4. Verdict rules

- Always against a reference checkpoint played with the same sides, seeds and match count; the
  reference is the best checkpoint by game match so far (`.claude/knowledge/INDEX.md`).
- A gap counts only beyond the match-to-match sd of both columns; one seed per training run ⟹
  write « une graine » next to any attribution to a code change.
- Open with what is worse than the reference. A checkpoint with more living tanks dry at 10 s or
  30 s than the reference is reported as worse on « reste dangereux », whatever its hits.
- Check every game label: the reading must rank labelled checkpoints as Samuel did. If not, stop
  and report the instrument as wrong.
- Trajectory: a run is judged on ≥ 3 saves, not on its last one.
- Before Samuel plays a save: give him the game-match table of that save and the prediction;
  after he plays, append his words verbatim to `.claude/knowledge/game_labels.md` the same day.

## 5. Write `<kdir>/matches/save_K.md`

`<kdir>` = the run folder of the knowledge base (CLAUDE.md); match exports stay in `<run>/matches/`.

English. Header (run, save, index, D, binary + commit, sides, matches), table against the
reference (bold best cell per row), what is worse first, prediction of the game verdict,
labels check. Short chat summary. Update `<kdir>/run.md` and the run's line in `INDEX.md`.
