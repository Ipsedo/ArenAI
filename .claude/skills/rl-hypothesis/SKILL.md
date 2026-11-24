---
name: rl-hypothesis
description: Mandatory method before proposing any cause or remedy for a learning problem of the ArenAI agents (PPO, PPO liquid, PPG, SAC) — the agent does not aim, trigger not selective, hit flat, entropy collapse, critic blind, curriculum stuck, regression after a peak. Turns the idea into a refutable hypothesis with a fast evaluator (probe, offline analysis, gradient/representation probe, oracle), runs that evaluator autonomously, and applies the gates for tactical trainings and confounded changes. Use for « pourquoi il n'apprend pas », « qu'est-ce qu'on essaie », « propose une solution », « quelle piste », « comment vérifier que X marche », and before suggesting any change to reward, network, loss, input, credit signal or hyperparameter.
---

# RL hypothesis → evaluator → remedy

Method: read `docs/rl_method.md` in full first. It defines symptom / cause / remedy / evaluator /
fast evaluator / tactical training / confound / noise floor / random floor and the invariants
M1–M6. This skill is the operational sequence.

## 1. Pin the symptom

- Latest facts on the run: `analyze-run` (CSV), `probe-timeline` or `probe-checkpoint` if probes
  exist. Use their written outputs if up to date.
- State the symptom with numbers, index, D, the random floor at that D and the reference run at
  equal D. Rule out artefacts first (window lag, in-sample `ev`, `fire` blind to the dump, `Hd`
  ≈ H(zoom), a column redefined between commits).

## 2. Load prior knowledge

`docs/decisions.md` (refused / lifted levers) + `MEMORY.md` + the notes of the run lineage. Extract
three lists and keep them in view: **refused levers**, **causes ruled out**, **intended designs**.
None of them is re-proposed without new evidence, and the evidence is then stated first.
Samuel says « on tourne en rond » ⟹ before any new proposal, name the circle: what was removed
and never restored, and which runs varied only what was left.

## 3. Candidate causes and the discriminating measurement

List candidates (bug, artefact, code/binary mismatch, curriculum, reward economy at the random
floor, credit horizon, exploration, optimisation budget, representation…). For each, the
measurement whose outcome differs from its competitors'. Pick by eliminations per unit of cost.
Instruments: `docs/rl_method.md` §Procedure 4.

## 4. Run the fast evaluators — no permission needed

Authorised without asking: launching `arenai_agent_probe` / `arenai_agent_sigma_gradients` and
the skills' scripts on existing checkpoints; writing new analysis scripts over existing exports;
adding a probe export or a new probe executable; building the binary a probe needs (worktree at
the run's effective commit when the inference code differs).

Constraints:
- no shared surface touched (else `design-gate`), nothing a training reads;
- a probe shares the GPU with a running training — launch in background, state the cost;
- new measurement code lives with the probes (`arenai_agent/probe/`) or with the skills' scripts
  (`.claude/skills/*/scripts/`), never inside trainer or environment classes (usage leak);
- validate a new instrument on a case with a known answer (`save_0` = random reference, a
  synthetic input, an oracle) before trusting it;
- report the verdict whatever it is: established / within noise / undecided, with counts.

## 5. Propose

Chat, French, conclusion first (shape in `docs/rl_method.md` §Output). Each proposal carries its
fast evaluator in full: metric, population, prediction if true, prediction if false, noise
floor, cost, and the result when already run. No evaluator ⟹ not a proposal; say what is
missing.

Form: the mechanism as LaTeX formulas (objective, estimator, gradient, expected value — symbols
defined once), the structure as a diagram (plug-in point, gradient flow and detach points,
causal chain). The same form applies to the technical detail that precedes a formulation
question.

Then the gates, each one explicit:

| situation | gate |
|---|---|
| remedy changes the problem formulation (network, head, loss, credit/advantage, input, reward, action space, curriculum rule) | technical detail first, then `AskUserQuestion`, **one question per piece** |
| remedy changes a default, an option, a hyperparameter default | `AskUserQuestion` |
| remedy changes a shared surface | `design-gate` |
| only a training can decide | `AskUserQuestion` with hypothesis, variable under test, reference run, decision rule (metric @ D/index ⟹ confirmed / refuted / undecided), stopping point, cost |
| the test is not measurable (several changes at once, no clean reference, metric redefined, binary mismatch) | **no button**: explain in chat what is confounded and what cannot be concluded, give the isolating alternative and its cost, ask Samuel to *type* that he imposes it; proceed only on that typed imposition and record it in the run's `analysis.md` |

## 6. After the answer

- Implement only the validated pieces. A piece implied but not asked is not validated.
- Before a tactical training starts: confirm the tree differs from the reference by the tested
  variable only (`git diff <reference effective commit> -- arenai_agent arenai_model arenai_core`),
  and name the checkpoints/indices at which the decision rule will be read.
- When the run has data: `analyze-run` / `probe-checkpoint` / `compare-runs`, apply the decision
  rule as written beforehand, and record the verdict — including refutation — in the run note.
