---
name: kaizen
description: Diagnose a miss — the gap between what Claude did and what it should have done — down to its cause in the harness, then hand the cause to `harness-improvement`. Use whenever Samuel says « kaizen », corrects how Claude works (« non pas comme ça », « pourquoi tu n'as pas… », « tu aurais dû », « encore une fois »), states a working rule or preference, or when Claude notices it broke a rule of CLAUDE.md, a skill or a doc. Not for a wrong result about the agent or the code (that is `rl-hypothesis` or an ordinary fix).
---

# Kaizen — diagnose the miss

Terms:
- **Miss**: one observable gap between Claude's action and the expected action. One kaizen = one
  miss; several misses ⟹ several diagnoses.
- **Expected action**: what a correct Claude would have done, stated as behaviour, not intent.
- **Harness**: everything that steers Claude in this repo — `CLAUDE.md`, `.claude/skills/*`
  (body and `description`), `docs/*.md`, hooks/settings in `.claude/`, memory. Memory is part of
  the harness only for facts about runs and project state; it never holds a working rule.
- **Cause class** (exactly one per miss):
  - `C1 absent` — no harness text covers the situation.
  - `C2 unrouted` — the rule exists but nothing loaded it at the moment of action (skill not
    triggered: `description` lacks the situation or the words; doc not referenced by the skill;
    rule buried in a file not loaded then).
  - `C3 ambiguous` — the rule was loaded but admits the wrong reading, or two rules conflict.
  - `C4 overridden` — the rule was loaded and clear, Claude followed a competing default
    (system prompt habit, generic tool guidance) instead. The fix makes the project rule win at
    that exact point.
  - `C5 misplaced` — the rule was written in a layer not read when it applies (e.g. a working
    rule saved to memory, a procedure in CLAUDE.md that only one skill needs).
  - `C6 executed wrong` — loaded, clear, not overridden, still not applied (slip). Fix = a check
    in the procedure that would have caught it, or a hook if it is mechanical.

## Procedure

1. **Quote the trigger.** Samuel's words verbatim (French), and the turn they refer to.
2. **State the miss** in one line each:
   - did: what Claude actually did (tool calls, text, file written);
   - expected: the expected action;
   - consequence: what it cost Samuel (time, wrong artefact, repeated correction).
3. **Locate the rule.** `grep -rn` the harness for the situation's key words (French and
   English). Record: path:line of every relevant rule, or « none ».
4. **Was it loaded?** From the transcript: which skill ran, which files were read before the
   action. A rule in an unread file was not loaded.
5. **Classify** into exactly one cause class C1–C6, with the evidence from steps 3–4. Prefer the
   most upstream class that is true (C1 before C2 before C3…).
6. **Undo the side effects** of the miss that Samuel rejected (e.g. delete a memory note that
   should be harness text), and say so.
7. **Report to Samuel** (French, conclusion first), in this shape:

   | | |
   |---|---|
   | déclencheur | « … » |
   | fait | … |
   | attendu | … |
   | règle existante | `path:line` ou aucune |
   | chargée au moment ? | oui / non (pourquoi) |
   | cause | C… — une phrase |

8. **Hand off**: invoke `harness-improvement` with the cause class and the evidence. A kaizen is
   not finished until the harness change is made (or proposed, when it touches Samuel's
   decisions — see that skill).

## Rules

- Diagnose the harness, not Claude's « attention ». « I forgot » is not a cause: find which
  layer failed to put the rule in front of Claude at that moment (C2/C5) or why it lost (C4).
- Never answer a kaizen only with a memory note.
- Never answer with an apology paragraph; the table is the answer.
