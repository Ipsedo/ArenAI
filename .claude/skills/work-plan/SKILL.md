---
name: work-plan
description: Write, apply and retire a work plan — a dated file in `.claude/plans/` that hands a decided direction (design, measurements, open pieces, steps) from one Claude session to the next. Plans are for Claude, never for users and never in `docs/`. Use whenever Samuel asks to write down what was decided so that another session codes it (« écris-moi ça », « écris le plan », « fais un plan », « je lancerai une nouvelle session pour le coder », « on a consommé trop de tokens »), whenever a session starts from a plan (« applique le plan », « code le plan panorama »), and at every commit of code that a plan describes (the plan file is deleted in that commit).
---

# Work plan — hand a decided direction to a later session

Terms:
- **Plan**: one file `.claude/plans/YYYY-MM-DD_<kebab-slug>.md`; date = the day it is written.
  Audience: Claude. English, dense, imperative; Samuel's verbatim quotes stay French.
- **Applied**: every step of the plan is coded, verified (`final-check`) and committed.

A plan is not documentation: `docs/` is public and describes what exists; a plan describes what
is to be done and disappears once done. A plan validates nothing: every gate (G1–G6) still
applies when the plan is applied, and the plan lists which pieces Samuel already validated
(date, his words) and which are still open.

## Write

1. `ls .claude/plans/` — extend an existing plan on the same subject rather than add a second.
2. Write the file with these sections, in this order, omitting none:
   - **Status**: date, what Samuel decided (verbatim), what is not decided.
   - **Why**: the symptom and the measurements that carry the direction, with run, save, index,
     D and the path of each script/CSV; refuted hypotheses included.
   - **Design**: the diagram (Mermaid, with its legend), formulas in LaTeX, every symbol defined.
   - **Validated pieces** / **Open pieces**: one line each; an open piece names its gate and the
     options to put to Samuel.
   - **Steps**: ordered, each with the files/area it touches and its verification.
   - **Evaluators**: the fast evaluators and the decision rule for the result, written before
     the code exists.
   - **Out of scope**: what the plan does not solve.
3. Self-sufficiency check: a session with no memory of this conversation must be able to start
   from the file alone. No « as discussed », no reference to the chat.
4. Ask for the commit (CLAUDE.md « Commits »), type `docs(plan)`.

## Apply

0. **The plan is committed before any of its code is written** (Samuel, 2026-11-27: « il doit
   commiter le plan avant de le coder »): `git status -- .claude/plans/` shows it untracked or
   modified ⟹ commit it first, alone (`docs(plan)`, CLAUDE.md « Commits »). Same after step 2
   changes it.
1. Read the plan in full, then verify every code fact it states against the current tree (a plan
   is a point-in-time note).
2. Put each open piece to Samuel through its gate before coding it; write his answer into the
   plan (move the line to **Validated pieces**).
3. Code step by step; a step that turns out wrong is corrected in the plan, not worked around.

## Retire

- The commit that completes a plan deletes its file: `git rm` the plan in the same commit as the
  code (Samuel, 2026-11-27: « lorsque le plan est appliqué et que son code est commité, on le
  supprime dans le même commit »). What must outlive it goes, in that commit too, to its durable
  place: `docs/decisions.md` (decision row), `docs/neural_architecture.md` /
  `docs/ARCHITECTURE.md` (what now exists), memory (facts about runs).
- A plan applied over several commits is deleted by the last one; earlier commits update its
  **Steps** (done steps removed).
- A plan Samuel abandons is deleted by a commit of its own, after the refusal is recorded in
  `docs/decisions.md`.
