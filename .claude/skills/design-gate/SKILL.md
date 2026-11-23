---
name: design-gate
description: Mandatory design procedure before writing C++ that adds, removes or changes any non-private element (method, constructor, parameter, field, free function, default argument) of a type or function referenced by two or more other classes/files, or that creates a new type meant to be shared — in include/ or src/. Also use on any doubt about whether a change is right in OOP, functional or modular-architecture terms, when tempted to add a getter, a setter, a flag or an extra parameter, and when Samuel asks for a design, a refactor or an API (« ajoute une méthode », « passe-lui X », « refactor », « comment exposer », « où mettre ça »).
---

# Design gate

Doctrine: read `docs/design_doctrine.md` in full first. It defines **shared surface**, the four
invariants I1–I4 (tell-don't-ask, constructor injection, no usage leak, minimal signatures) and
the judgement criteria. This skill is the procedure that applies it.

## 1. Measure the surface

For every type or free function the change would touch or create:

```shell
grep -rn --include='*.h' --include='*.cpp' -w '<TypeOrFunction>' arenai_*/ | grep -v '/tests/'
```

Count distinct classes/files other than the owner. ≥ 2 ⟹ shared surface. A new type that will be
used from ≥ 2 places ⟹ shared surface. Tests count as users when deciding the blast radius, not
when deciding whether the surface is shared. Also list the siblings (other agents, other
distributions, other metrics) that mirror the type: their symmetry is part of the question.

## 2. Find the owner, then design

Follow the decision procedure of the doctrine (owner → collaborator vs operand → ≥ 2 real designs
→ I1–I4 → ranking → cost). Write for yourself, before any question:

- the need, in one sentence, in domain terms;
- the owner of the state or knowledge involved;
- each surviving design with its **exact signatures** (declaration lines as they would appear in
  the header) and the call sites it rewrites;
- which invariant killed each discarded design.

If the only designs found violate an invariant, say so in the question instead of presenting
them as acceptable.

## 3. Ask

`AskUserQuestion`, in French, before any code:

- one question per independent choice (never a bundled "do you validate?");
- 2–3 options, recommended first, label = the design's name, description = why and its cost
  (files, call sites, siblings to realign);
- `preview` = the proposed declarations, verbatim;
- the question text stands alone: name the class, the need, what currently exists;
- when the choice is about collaboration between several classes, draw it in the chat first
  (class/collaboration diagram: who owns, who is injected into whom, dependency direction).

Not shared, no doubt, no contract or behaviour change ⟹ no question; proceed.

## 4. Implement exactly what was validated

- The validated signature, character for character. A needed deviation is a new question.
- A second shared surface discovered on the way is a new question. Approval does not propagate.
- Realign siblings only if that was part of the validated design.
- Comments: one line at most, only to disambiguate (shape, unit, index) or to title a section.
- Finish with the `final-check` skill.

## 5. Self-review before declaring done

Re-read the diff as Samuel would:

- any `get_*`/`set_*` introduced? any accessor whose caller computes on the result? (I1)
- any parameter that every call site passes identically, or that is threaded through? (I2)
- any member that dies if one caller is deleted? (I3)
- any new `bool`, default argument, optional, `if (mode)`? (I4)
- any output parameter, widened ownership, upward dependency, forwarding-only layer?

One "yes" ⟹ not done: fix within the validated design, or ask.
