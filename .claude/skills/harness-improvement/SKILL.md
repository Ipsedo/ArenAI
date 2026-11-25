---
name: harness-improvement
description: Diagnose a miss — the gap between what Claude did and what it should have done — down to its root cause in the harness, then change the harness (CLAUDE.md, .claude/skills, docs/, hooks/settings, memory) so that it cannot recur. Use whenever Samuel corrects how Claude works (« non pas comme ça », « pourquoi tu n'as pas… », « tu aurais dû », « encore une fois »), states a working rule or preference, when Claude notices it broke a rule of CLAUDE.md, a skill or a doc, when Samuel asks to put a rule « dans le harnais », to « améliorer le harnais », to create or edit a skill, for each row Samuel selected from a `kaizen` table, or when a working rule has to be recorded anywhere. Never record a working rule in memory instead. Not for a wrong result about the agent or the code (that is `rl-hypothesis` or an ordinary fix).
---

# Harness improvement — diagnose a miss, remove its cause

Input, one of:
- a **miss** (correction from Samuel, rule Claude broke) ⟹ full procedure from step 1;
- a **stated rule** or a harness request with no miss behind it ⟹ step 1 reduces to 1.3–1.5
  (locate, loaded?, classify — usually `C1`, or `C3` when it sharpens an existing rule);
- a **`kaizen` row** Samuel selected (waste, evidence, current standard, small step) ⟹ the
  waste is the miss, its evidence stands for 1.1–1.2; run 1.3–1.5, no second report (1.7);
- rules already written in the wrong layer (e.g. memory notes of `type: feedback`) ⟹ a batch
  of `C5 misplaced`, the procedure once per rule.

Terms:
- **Miss**: one observable gap between Claude's action and the expected action. One diagnosis =
  one miss; several misses ⟹ several diagnoses.
- **Expected action**: what a correct Claude would have done, stated as behaviour, not intent.
- **Harness**: everything that steers Claude in this repo — `CLAUDE.md`, `.claude/skills/*`
  (body and `description`), `.claude/agents/*`, `docs/*.md`, hooks/settings in `.claude/`,
  memory. Memory is part of the harness only for facts about runs and project state; it never
  holds a working rule.
- **Cause class** (exactly one per miss):
  - `C1 absent` — no harness text covers the situation.
  - `C2 unrouted` — the rule exists but nothing loaded it at the moment of action (skill not
    triggered: `description` lacks the situation or the words; doc not referenced by the skill;
    rule buried in a file not loaded then).
  - `C3 ambiguous` — the rule was loaded but admits the wrong reading, or two rules conflict.
  - `C4 overridden` — the rule was loaded and clear, Claude followed a competing default
    (system prompt habit, generic tool guidance) instead.
  - `C5 misplaced` — the rule was written in a layer not read when it applies (e.g. a working
    rule saved to memory, a procedure in CLAUDE.md that only one skill needs).
  - `C6 executed wrong` — loaded, clear, not overridden, still not applied (slip).

## Philosophy (binding)

- **Audience is Claude.** Write in the wording that steers Claude best: English, dense,
  imperative, strong defined terms, no prose for Samuel. Samuel-facing output stays French.
- **M6 applies to the harness.** Remove the cause the diagnosis named; never add a
  « remember to … » reminder next to a rule Claude already had (that patches C2–C5 with C1 text).
- **One rule, one place.** Before writing, `grep -rn` for an existing rule on the same subject;
  extend or sharpen it rather than duplicate. A rule stated twice drifts.
- **Put the rule where it is read at the moment of action.** The layer is chosen by when the
  rule must be in front of Claude, not by importance.
- **Smallest sufficient text.** Every line in an always-loaded file costs every turn.

## Layer selection

| the rule applies… | layer | form |
|---|---|---|
| to every task / every answer | `CLAUDE.md`, the section it belongs to | one bullet, imperative |
| only inside one procedure | that skill's `SKILL.md` body | a step or a rule in its list |
| it was not loaded because the situation did not trigger the skill (C2) | that skill's `description` | add the situation and Samuel's literal words (French) |
| a new recurring situation with its own procedure (≥ 3 steps) | new skill `.claude/skills/<name>/SKILL.md` + a row in CLAUDE.md « Skill routing » | frontmatter `name`, `description` with triggers |
| Samuel refuses or lifts a lever / design | `docs/decisions.md` | one dated row, reason and scope — public file: no name, no conversation quote |
| normative reference several skills load | `docs/<topic>.md`, referenced from each skill that needs it | defined terms + invariants |
| mechanical, checkable without judgement (format, forbidden path, command) | hook or script in `.claude/` (settings via `update-config`) | enforcement, not text |
| a fact about runs, results, project state | memory | never a working rule |

Cause class → default fix:
- `C1 absent` — write the rule in the layer above.
- `C2 unrouted` — fix routing: skill `description` triggers, CLAUDE.md routing row, or the
  skill's « read X first » line. Do not copy the rule elsewhere.
- `C3 ambiguous` — rewrite the rule with a defined term or a decision table; delete the
  conflicting variant.
- `C4 overridden` — state explicitly that the project rule overrides the named default, at the
  point where the default fires.
- `C5 misplaced` — move the text (delete from the wrong layer, write in the right one).
- `C6 executed wrong` — add a verification step to the procedure, or a hook if mechanical.

## Procedure

1. **Root-cause diagnosis** — before choosing any layer or writing any text.
   1. **Quote the trigger.** Samuel's words verbatim (French), and the turn they refer to.
   2. **State the miss**, one line each — did: what Claude actually did (tool calls, text, file
      written); expected: the expected action; consequence: what it cost Samuel (time, wrong
      artefact, repeated correction).
   3. **Locate the rule.** `grep -rn` the harness for the situation's key words (French and
      English). Record `path:line` of every relevant rule, or « none ».
   4. **Was it loaded?** From the transcript: which skill ran, which files were read before the
      action. A rule in an unread file was not loaded.
   5. **Classify** into exactly one cause class C1–C6, with the evidence from 1.3–1.4. Prefer
      the most upstream class that is true (C1 before C2 before C3…).
   6. **Undo the side effects** of the miss that Samuel rejected (e.g. delete a memory note
      that should be harness text), and say so.
   7. **Report the diagnosis** to Samuel (French, conclusion first), in this shape:

      | | |
      |---|---|
      | déclencheur | « … » |
      | fait | … |
      | attendu | … |
      | règle existante | `path:line` ou aucune |
      | chargée au moment ? | oui / non (pourquoi) |
      | cause | C… — une phrase |

2. Choose the layer (« Layer selection ») from the cause class, and read its files around the
   insertion point (and CLAUDE.md « Skill routing » when a skill changes).
3. Write the change. Skill frontmatter: `name` = folder name; `description` = what it does +
   « Use whenever … » with situations and Samuel's literal phrasings.
4. **Replay check**: re-read the triggering situation; list what would now be loaded, in which
   order, and confirm it yields the expected action. If not, the layer is wrong — redo.
5. **Collision check**: `grep -rn` for rules the change contradicts or duplicates, in the harness
   **and** in memory; then per hit:
   - same rule → keep one copy in the chosen layer, merge any extra precision (scope, date,
     Samuel's verbatim) into it, delete the other copy;
   - stricter or narrower variant → merge as the precise form, delete the vague one;
   - contradiction → do not pick: `AskUserQuestion` quoting both texts with their path and date.
6. Clean up: delete memory notes or stray text the miss produced.
7. Report to Samuel in French: one line per file changed (path + what) and the replay result;
   then ask for the commit (CLAUDE.md « Commits »).

## Rules

- Diagnose the harness, not Claude's « attention ». « I forgot » is not a cause: find which
  layer failed to put the rule in front of Claude at that moment (C2/C5) or why it lost (C4).
- Never answer a miss only with a memory note, nor with an apology paragraph: the diagnosis
  table and the harness change are the answer.
- A miss is not closed until the harness change is made (or proposed, when it touches Samuel's
  decisions — « Authority »).

## Authority

- Free: harness edits that fix a diagnosed miss, implement a `kaizen` row Samuel selected or a
  rule Samuel stated (wording, routing,
  placement, new procedural skill he asked for).
- Ask first (`AskUserQuestion`): any change to the gates G1–G6, to Samuel's authority, to what
  counts as free work, or deleting a rule Samuel wrote.
