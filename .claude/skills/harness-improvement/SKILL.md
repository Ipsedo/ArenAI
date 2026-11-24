---
name: harness-improvement
description: Change the harness (CLAUDE.md, .claude/skills, docs/, hooks/settings, memory) so that a diagnosed miss cannot recur. Use right after `kaizen`, when Samuel asks to put a rule « dans le harnais », to « améliorer le harnais », to create or edit a skill, or when a working rule has to be recorded anywhere. Never record a working rule in memory instead.
---

# Harness improvement — remove the cause of a miss

Input: a `kaizen` diagnosis (miss, expected action, cause class C1–C6, evidence). Without one,
run `kaizen` first. Exception: migrating rules already written in the wrong layer (e.g. memory
notes of `type: feedback`) is a batch of `C5 misplaced` — run the procedure once per rule.

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
| Samuel refuses or lifts a lever / design | `docs/decisions.md` | one dated row, his words, scope |
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

1. Read the files of the chosen layer around the insertion point (and CLAUDE.md « Skill
   routing » when a skill changes).
2. Write the change. Skill frontmatter: `name` = folder name; `description` = what it does +
   « Use whenever … » with situations and Samuel's literal phrasings.
3. **Replay check**: re-read the triggering situation; list what would now be loaded, in which
   order, and confirm it yields the expected action. If not, the layer is wrong — redo.
4. **Collision check**: `grep -rn` for rules the change contradicts or duplicates, in the harness
   **and** in memory; then per hit:
   - same rule → keep one copy in the chosen layer, merge any extra precision (scope, date,
     Samuel's verbatim) into it, delete the other copy;
   - stricter or narrower variant → merge as the precise form, delete the vague one;
   - contradiction → do not pick: `AskUserQuestion` quoting both texts with their path and date.
5. Clean up: delete memory notes or stray text the miss produced.
6. Report to Samuel in French: one line per file changed (path + what), the replay result, and
   « non commité ». Commit only on request.

## Authority

- Free: harness edits that implement a kaizen or a rule Samuel stated (wording, routing,
  placement, new procedural skill he asked for).
- Ask first (`AskUserQuestion`): any change to the gates G1–G6, to Samuel's authority, to what
  counts as free work, or deleting a rule Samuel wrote.
