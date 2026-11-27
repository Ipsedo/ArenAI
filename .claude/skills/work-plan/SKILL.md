---
name: work-plan
description: Mandatory four-phase workflow of every Feature (CLAUDE.md « Feature ⟹ plan ») — draft a plan in `.claude/plans/` as soon as Samuel validated the idea, put its pieces through the gates, get the plan validated and committed, execute it, get the result validated and committed (the plan file is deleted in that commit). Also hands a decided direction from one Claude session to the next. Plans are for Claude, never for users and never in `docs/`. Use whenever Samuel validates the idea of a new feature (« ok on fait ça », « go », « on part là-dessus », « code-le », « implémente », « ajoute X »), before writing any code of it; whenever Samuel asks to write down what was decided (« écris-moi ça », « écris le plan », « fais un plan », « je lancerai une nouvelle session pour le coder », « on a consommé trop de tokens »); whenever a session starts from a plan (« applique le plan », « code le plan panorama »); and at every commit of code that a plan describes.
---

# Work plan — draft, validate, execute, close

Terms:
- **Feature**: defined in CLAUDE.md « Feature ⟹ plan » (with what is not one). Doubt ⟹ it is a
  Feature.
- **Plan**: one file `.claude/plans/YYYY-MM-DD_<kebab-slug>.md`; date = the day it is written.
  Audience: Claude. English, dense, imperative; Samuel's verbatim quotes stay French.
- **Piece**: one independent choice of the Feature that a gate (G1–G6) puts to Samuel.
- **Validated plan**: every piece answered through its gate, or listed open with the code fact
  it waits for; Samuel answered phase 2; the file is committed.
- **Applied**: every step coded, verified (`final-check`), evaluators run, committed.

A plan is not documentation: `docs/` is public and describes what exists; a plan describes what
is to be done and disappears once done. A plan validates nothing by itself: a piece is validated
only by Samuel's answer to its own gate question, recorded in the plan (date, his words).

Phases, in order, none skipped. **No code of the Feature is written before phase 3.** Reading
code, running fast evaluators and disposable smoke checks are allowed in every phase.

## 1. Draft — on Samuel's validation of the idea

1. `ls .claude/plans/` — extend an existing plan on the same subject rather than add a second.
2. Write the file with these sections, in this order, omitting none:
   - **Status**: `draft` | `validated` | `in progress`; date; what Samuel decided (verbatim);
     what is not decided.
   - **Why**: the symptom and the measurements that carry the direction, with run, save, index,
     D and the path of each script/CSV; refuted hypotheses included.
   - **Design**: the diagram (Mermaid, with its legend), formulas in LaTeX, every symbol defined;
     exact signatures of every shared surface touched.
   - **Validated pieces** / **Open pieces**: one line each; an open piece names its gate and the
     options to put to Samuel.
   - **Steps**: ordered, each with the files/area it touches and its verification; public docs
     to update (`docs/ARCHITECTURE.md`, `BUILD.md`, `USAGE.md`) are steps.
   - **Evaluators**: the fast evaluators and the decision rule for the result, written before
     the code exists (M1); a tactical training is a G4 piece.
   - **Out of scope**: what the plan does not solve.
3. Run the harness on the draft: formulation ⟹ skill `rl-hypothesis` (M1–M6, prior knowledge in
   `docs/decisions.md`); shared surface or design doubt ⟹ skill `design-gate`; then put every
   open piece to Samuel through its gate (`AskUserQuestion`, one question per piece, CLAUDE.md
   « Gates »). Write each answer into the plan and move its line to **Validated pieces**.
   A piece that cannot be asked before some code exists stays open, with that reason.
4. Self-sufficiency check: a session with no memory of this conversation must be able to start
   from the file alone. No « as discussed », no reference to the chat.

## 2. Validate the plan — Samuel

1. Chat, French, CLAUDE.md « Answers to Samuel »: the plan's diagram, then the summary — what
   will be built, pieces validated, pieces still open, steps, evaluators — and the link to the
   file.
2. One `AskUserQuestion` call, two questions:
   - commit the plan? — shows the message (`docs(plan): …`); options: commit / correct the plan;
   - where to execute? — in this session / end the session here (a later session applies it).
   This answer validates the plan as a faithful record and its step order, never a piece.
3. Corrections ⟹ back to phase 1.3. Commit ⟹ set **Status** `validated`, commit the plan alone
   (Samuel, 2026-11-27: « il doit commiter le plan avant de le coder »).
4. End of session chosen ⟹ stop; give the one-line prompt that starts phase 3 (« applique le
   plan <slug> »).

## 3. Execute

0. `git status -- .claude/plans/` shows the plan untracked or modified ⟹ it is not a validated
   plan: phase 2 first.
1. Read the plan in full, then verify every code fact it states against the current tree (a plan
   is a point-in-time note).
2. Put each remaining open piece to Samuel through its gate before coding it; write his answer
   into the plan and commit that change (`docs(plan)`) before the code.
3. Code step by step, exactly the validated pieces. A step that turns out wrong is corrected in
   the plan, not worked around; a new piece discovered on the way is a new gate question.
4. Finish: skill `final-check`, then run the plan's **Evaluators** and apply their decision rule.
5. Move what must outlive the plan to its durable place: `docs/decisions.md` (decision row),
   `docs/neural_architecture.md` / `docs/ARCHITECTURE.md` / `BUILD.md` / `USAGE.md` (what now
   exists), knowledge base (facts about runs and project state).

## 4. Close — Samuel

1. Chat, French: what was built against the plan — steps done, deviations and why, evaluator
   verdicts, what is not done or not verified.
2. Ask for the commit (CLAUDE.md « Commits »). The commit that completes the plan deletes its
   file: `git rm` the plan in the same commit as the code (Samuel, 2026-11-27: « lorsque le plan
   est appliqué et que son code est commité, on le supprime dans le même commit »).
- A plan applied over several commits is deleted by the last one; earlier commits update its
  **Steps** (done steps removed) and set **Status** `in progress`.
- A plan Samuel abandons is deleted by a commit of its own, after the refusal is recorded in
  `docs/decisions.md`.

## Handoff without a Feature

Samuel asks to write down a decided direction (« écris le plan », « on a consommé trop de
tokens ») for something that is not a Feature ⟹ same file, same phases; phase 1 starts at his
request instead of at an idea validation.
