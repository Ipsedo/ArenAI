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

## Procedure

1. **Symptom.** State it with numbers, index, D, and its distance to the random floor and to the
   reference run at equal D. Check it is not a measurement artefact (window lag, column
   redefinition, in-sample metric, masked dimension).
2. **Prior knowledge.** Read `MEMORY.md` and the linked run notes. A lever recorded as refused
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
8. **Record.** A fact established about a run goes into that run's memory note, phrased as fact
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
