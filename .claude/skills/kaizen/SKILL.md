---
name: kaizen
description: Continuous improvement of the harness (改善) — find waste in how Claude works in this repo, without waiting for a miss, and propose small improvement steps Samuel selects; the selected steps go to `harness-improvement`. Use whenever Samuel says « kaizen » (with or without a scope — « kaizen sur les sondes », « kaizen sur ce qu'on vient de faire »), « amélioration continue », « qu'est-ce qu'on peut améliorer dans ta façon de travailler », and on Claude's own initiative as soon as it observes waste in its own work (step redone, avoidable question, repeated manual command, rule loaded for nothing, stale or duplicate harness text). Not for a correction from Samuel, a stated working rule or a broken rule — that is a miss, go straight to `harness-improvement`. Not for the code or the agent.
---

# Kaizen — continuous improvement of the harness

A miss is not required: the current standard works and can still be made better. A miss
(correction from Samuel, stated rule, broken rule) is not a kaizen ⟹ `harness-improvement`.

Terms:
- **Harness**: everything that steers Claude in this repo — `CLAUDE.md`, `.claude/skills/*`
  (body and `description`), `.claude/agents/*`, normative `docs/*.md`, hooks/settings in
  `.claude/`, memory. The only scope of a kaizen: never the game, the agent or the tooling code.
- **Standard**: the harness text or hook that currently governs a situation. Every improvement
  starts from the standard and ends as a new standard; no standard ⟹ writing one is the step.
- **Waste** (muda) — work that brought Samuel nothing:
  - `W1 rework` — a step redone, an artefact rewritten, a build or probe relaunched for a
    reason the harness could have stated upfront;
  - `W2 avoidable question` — a question to Samuel whose answer was already in the harness, the
    code or a previous answer; or a missing question that cost a round trip;
  - `W3 manual repetition` — the same commands or checks typed by hand across tasks, that a
    script, a hook or a skill step would carry;
  - `W4 overload` — text loaded every turn (CLAUDE.md, skill descriptions, memory index) that
    rarely changes an action; a skill loading more than its procedure needs;
  - `W5 stale standard` — a rule, path, command, default or memory note that no longer matches
    the repo; a duplicated rule; two rules that drifted apart;
  - `W6 detour` — a correct result reached by a longer path than the harness could have
    pointed to (wrong skill first, search for a known location, output Samuel had to trim).
- **Gemba**: the place where the work happened — this transcript, the files, the git log, the
  run folders. A waste is recorded only with evidence from there (turn, `path:line`, command);
  a waste imagined from the armchair is not a row.
- **Small step**: one improvement = one change in one layer, reversible, checkable on the next
  occurrence. A large idea is split into steps or dropped.

## Scope of one kaizen

- Samuel names a scope (« kaizen sur X ») ⟹ that scope: the skill, doc, procedure or stretch of
  work he names.
- « kaizen » alone ⟹ the work of this session.
- Own initiative ⟹ the single waste just observed. Finish the atomic step in progress (never
  leave a build, an edit or a measurement half-done), run this procedure on that waste, then
  resume the task.

## Procedure — PDCA

1. **Go to the gemba.** Re-read the scope: transcript turns, the skills that ran, the harness
   files that were loaded and those that should have been. `grep -rn` the harness for the
   standard governing each situation.
2. **List the wastes**, each with its class W1–W6, its evidence and the current standard
   (`path:line` or « none »). Drop any without evidence.
3. **Plan**: for each waste, one small step, and the gain expected at the next occurrence,
   stated so that it can be checked (« the probe binary check runs before the build », not
   « smoother »). No step that touches a gate G1–G6, Samuel's authority or what counts as free
   work without saying so in the row.
4. **Report to Samuel** (French, conclusion first), one table, most valuable row first:

   | # | gaspillage | preuve | standard actuel | petit pas | gain vérifiable |
   |---|---|---|---|---|---|

   No waste found ⟹ say so in one line; never pad the table.
5. **Samuel selects**: `AskUserQuestion`, multiSelect over the rows (> 4 rows ⟹ options =
   groups of rows by harness file). Nothing is changed before his answer.
6. **Do**: invoke `harness-improvement` once per selected row, passing the row as its input.
7. **Check**: that skill's replay check is the immediate check. The real one is the next
   occurrence: when the situation returns, note in one line whether the gain held.
8. **Act**: gain held ⟹ the change is the standard, and the baseline of the next kaizen. Gain
   absent ⟹ revert the step and say so; a step kept without its gain is itself W5.

## Rules

- Improve the system, never Claude's « attention »: a row whose step is « be more careful » is
  not a row.
- Small steps, often: prefer three one-line rows over one restructuring.
- Removing text is an improvement as much as adding it; W4 and W5 rows usually delete.
- Never answer a kaizen with a memory note.
