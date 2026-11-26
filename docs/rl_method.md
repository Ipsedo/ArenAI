# RL problem-solving method

Audience: Claude. Normative. Loaded by the `rl-hypothesis` skill and by every run/probe analysis
skill. Goal: every claim about why the agent fails, and every remedy, is tied to a measurement
that can refute it, at the lowest cost that can refute it.

## Vocabulary

- **Symptom**: an observed metric behaviour (CSV column, probe metric), with its index and D.
- **Cause**: a mechanism in the code or the problem formulation that produces the symptom.
  A cause is named as a causal chain, each link observable.
- **Hypothesis**: `cause ⟹ symptom`, stated so that a measurement can contradict it.
- **Remedy**: a change that removes the cause. A change that only moves the symptom is a
  **patch** and is forbidden except as an announced, disposable diagnostic.
- **Evaluator**: the measurement that decides a hypothesis or a remedy. Defined by: metric,
  population (which decisions/episodes/checkpoints), prediction if true, prediction if false,
  noise floor, cost.
- **Fast evaluator**: an evaluator that needs no training — a probe on existing checkpoints, an
  offline analysis of existing CSV/probe exports, a gradient or representation probe, a unit test
  on a synthetic input, an oracle/ablation replayed at inference.
- **Tactical training**: a training run launched to prove or refute a remedy, whatever its
  length.
- **Confound**: two or more differences between the compared conditions, so the effect cannot be
  attributed. Sources: several code changes, code + config, different binary, different
  `nb_tanks`, a metric whose definition changed, uncommitted code, a reused run folder.
- **Noise floor**: variability between identical configurations (n = 1 seed per run). Known
  orders of magnitude: curriculum milestone D ±30k index, emptied reserve ×2, hit/shell ±1 point,
  probe ratio unreliable under ~300 aligned-and-close decisions.
- **Lever**: a change whose effect is established on a **game outcome** (M7). A gap on a
  train-rule metric (CSV `kill`/`hit`/`fire`, probe hits/tank-s, `tour<5°`, selectivity) or on an
  intermediate indicator (sign of μ, a correlation, a corrector gain) is reported as « indicator,
  game effect not established », never as a lever, never as "better".
- **Train rules / game rules**: the training and probe environment (`TrainTankEnvironment`)
  refills a dry tank (respawn at empty reserve), cuts episodes at `max_episode_steps`, samples
  actions, runs 32 tanks without a player. The game (`DesktopGameEnvironment` + `GameSession`)
  never refills, has no time limit, acts deterministically, runs `nb_tanks` 16 with the player's
  tank in the arena, `GAME_SHELL_ECONOMY`. Any count that includes actions after a refill does
  not exist in game.
- **Game match**: the measurement under game rules — `arenai_agent_match` (skill `game-match`).
- **Game outcome**: a game-match metric of Samuel's criteria (2026-11-25, his order): « trouve les
  ennemis », « reste dangereux longtemps », « touche le joueur », « touche et tue les autres
  tanks », « tenue visible ».
- **Game label**: Samuel's own verdict after playing a checkpoint. The only ground truth. Recorded
  in `.claude/knowledge/game_labels.md` (checkpoint, date, his words).
- **Random floor**: the metric value of an untrained or random policy at the same D. Always the
  reference for "has it learned".
- **Structural effect**: imposed mechanically by the code; established by reading the code.
  Everything else must beat the noise floor.

## Invariants

### M1 — No proposal without its fast evaluator

Every proposed cause and every proposed remedy ships with a fast evaluator, written out in full
(metric, population, both predictions, noise floor, cost). A proposal whose evaluator cannot be
stated is not ready to be presented; say what is missing instead.

### M2 — Run the fast evaluator before proposing the remedy

Fast evaluators are within Claude's authority (validated 2026-11-23): launch existing probes and
scripts on existing checkpoints, write new measurements (analysis scripts, probe exports), build
the binary a probe needs. No question needed. Limits:

- the new measurement touches no shared surface (else `docs/design_doctrine.md` gate) and no
  training code path;
- it changes nothing that a running or future training would see;
- it is reported with its cost and its result, including when it refutes Claude's own
  hypothesis.

Present the hypothesis *with* the evaluator's verdict whenever it can be obtained first.

### M3 — Tactical training is Samuel's decision

Any training run, however short, is asked through `AskUserQuestion` before launch. The question
carries: the hypothesis, what the fast evaluators already established, the single variable under
test, the reference run, the decision rule (which metric at which D/index means confirmed /
refuted / undecided), the stopping point, the cost in wall-clock and GPU.

### M4 — One variable, or an explicit imposition

A test must isolate one variable against a reference that differs by that variable only. If the
planned change is not measurable — several changes at once, no valid reference, metric
redefined, binary mismatch — do not proceed and do not ask through a button. Write in the chat:
what will be confounded, what conclusion becomes impossible, the isolating alternative and its
cost; then ask Samuel to **type in the chat that he imposes it**. Only a typed, explicit
imposition unlocks it. Record the imposition and the lost attribution in the run's analysis.

### M5 — Formulation changes are validated piece by piece

Anything that changes how the problem is posed — network, encoder, head, loss, credit or
advantage signal, input, reward, action space, curriculum rule — is detailed first (where it
plugs in, what is learned, on which data, which gradients flow where, cost) and then asked one
explicit question per piece. Approval of a named direction validates the principle only, never
the architectural pieces it implies. A global "do you validate?" does not count.

### M6 — Remove causes

Before any remedy: name the cause, say how the remedy removes it, and if it does not, say so.
Floors, ceilings, thresholds, coefficients and special cases added to compensate are patches.

### M7 — Only the game judges a run (2026-11-25)

Trigger: Samuel judged 553 s36 « pourri en jeu » after Claude called it the best aim of three
runs from train-rule kill and `tour<5°`; over half of its hits came after a free refill.

- The words good / better / best / improved / « meilleur », « bon », « progrès » about a run or a
  checkpoint are allowed only on game outcomes measured by a game match, against a reference
  checkpoint measured the same way, beyond the match-to-match noise. Otherwise write
  « indicateur (règles d'entraînement), effet en jeu non mesuré ».
- A game-match result is a **prediction** of Samuel's verdict, stated as such. The verdict is his.
- Every game label is recorded the day it is given; the game-match reading must rank all labelled
  checkpoints as Samuel did. A label it misranks ⟹ the instrument is wrong: say so, fix the
  instrument before any further verdict.
- Bad news first: any game outcome worse than the reference (reserve emptied, disarmed share,
  player never hit) opens the analysis, before any favourable indicator.
- Until a game match exists for a checkpoint, the only admissible proxy from an existing probe
  is the **first-life reading** (events before each tank's first empty reserve), labelled as an
  approximation.

## Procedure

1. **Symptom.** State it with numbers, index, D, and its distance to the random floor and to the
   reference run at equal D. Check it is not a measurement artefact (window lag, column
   redefinition, in-sample metric, masked dimension).
2. **Prior knowledge.** Read `.claude/knowledge/INDEX.md` and the linked run notes. A lever recorded as refused
   is not re-proposed. A cause recorded as ruled out (e.g. CNN perception) is not re-proposed
   without new evidence. A design recorded as intended (e.g. aim as a potential) is not
   challenged from the fix.
3. **Candidate causes.** List every mechanism that would produce the symptom, including the
   boring ones: bug, metric artefact, binary/code mismatch, curriculum effect, economy of the
   reward at the random floor. Rank by prior and by cost to refute.
4. **Discriminating evaluator.** For each candidate, find the measurement whose outcome differs
   between this cause and its competitors. Prefer the one that eliminates the most candidates per
   unit of cost. Typical instruments:
   - behaviour by error bucket (aligned / off-axis, near / far), never a global mean;
   - correlation of the action mean with the signed error, by error bucket, good sign stated;
   - `∂Q/∂a` or advantage sign agreement with the oracle direction;
   - linear probe (R²) of a geometric quantity on the encoder features, with `save_0` as
     reference;
   - trigger selectivity (`p_al` / `p_off`), dump, emptied reserve;
   - reward decomposition per event, expected value of one more shot at the random floor;
   - oracle substitution at inference (replace one head by the geometric answer) to bound what a
     perfect version of that piece would yield;
   - trajectory across saves, never a single checkpoint.
5. **Run it** (M2). Report: established / within noise / undecided, with numbers and counts.
6. **Remedy.** Only for a cause that survived. State how it removes the cause (M6), which
   formulation pieces it touches (M5), and its own fast evaluator (M1) — ideally the same
   instrument that exposed the cause, with the predicted change.
7. **Escalate if needed.** If only a training can decide, M3. If it cannot be isolated, M4.
8. **Record.** A fact established about a run goes into that run's `run.md` (knowledge base, CLAUDE.md), phrased as fact
   + evidence + scope. Refuted hypotheses are recorded too.

## Reading discipline

- Compare at equal D first, equal index second; tank-steps or wall-clock when `nb_tanks`
  differs. Never beyond the D reached by the least advanced run.
- Judge aim and trigger on hit/fire, selectivity and probe metrics, not on `r` across reward
  definitions.
- A single record proves nothing. Require an effect held over several D levels or ≥ 3 saves.
- An identical curriculum calendar between runs proves nothing (seeded curriculum).
- The code that matters is the code *of the run* (effective commit, binary mtime), not the
  working tree.
- Every quantitative statement carries its index and D.
- **Cruise reading (D = 1000)**, in this order: game outcomes from a game match (M7); then, as
  indicators only, hits/tank-s per distance band (probe, sides 550 and 1000) in first life, dump
  (emptied reserve: median, share emptying, share disarmed), hit/shell, trigger selectivity
  (`p_al` / `p_off`); CSV kill/hit, D, fire, Hc/σ only to explain.

## Output shape for a proposal (chat, French, concise)

Mathematics is written in LaTeX (`$…$`, `$$…$$`), every symbol defined once: the objective, the
estimator, the gradient path, the expected value that makes the argument. Structure is drawn
(visualize widget when available, else Mermaid): where the piece plugs into the network, where
gradients flow and where they are detached, the causal chain from cause to symptom. Never a
formula or an architecture paraphrased in prose.

1. Conclusion in one sentence.
2. Cause, as a chain, with the evidence already measured.
3. Remedy and how it removes the cause; pieces it changes.
4. Fast evaluator: metric, population, prediction if true / if false, noise floor, cost — and
   its result if already run.
5. What is still undecided, and the gate that applies (M3 question, M4 typed imposition, M5
   per-piece questions).
