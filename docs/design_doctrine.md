# Design doctrine

Audience: Claude. Normative. Loaded by the `design-gate` skill and whenever a design choice is in
play. Samuel judges code on its beauty: object design, functional purity, modular architecture.
A change that works but degrades the design is a rejected change.

## Vocabulary

- **Shared surface**: every non-private element of a type or free function that is referenced by
  two or more other classes or files, wherever it lives (`*/include/` or `src/`). Elements:
  public and protected members, constructors, free functions, struct fields, type aliases,
  template parameters, virtual interface, default arguments.
- **Owner**: the class that holds a piece of state. Behaviour over that state belongs to it.
- **Collaborator**: an object a class needs for its whole life. **Operand**: a value a single call
  works on.
- **Usage leak**: anything in a class that exists only because one particular caller needs it
  (probe, test, training loop, GUI, one agent).
- **Patch**: an ad hoc element (flag, default, special case, extra accessor) whose only
  justification is avoiding a redesign.

## Invariants (non-negotiable, validated by Samuel 2026-11-23)

A proposal that violates one of these is not an option to present. Redesign until none is
violated, or state explicitly that no compliant design was found and ask.

### I1 — Tell, don't ask

No accessor that exports internal state so that a caller can compute with it. Move the
computation into the owner and expose the *result* or the *behaviour*.

- Smell: `get_x()` followed, at the call site, by arithmetic or branching on `x`.
- Smell: two callers repeating the same derivation from the same getter.
- Legitimate: returning a value that *is* the class's product (a network's output, a metric's
  value, an immutable value object handed over whole).
- Test: "would the caller break if the owner changed its internal representation?" Yes ⟹
  violation.

### I2 — Constructor injection

A collaborator or a configuration value that is stable for the object's life enters through the
constructor and is stored as a member. Never through a parameter added to a method, never
through a setter called after construction, never through a global.

- Smell: a method parameter that every call site passes with the same value.
- Smell: a parameter threaded through N intermediate signatures to reach its real consumer.
- Method parameters are operands only.
- If the constructor grows unreadable, the answer is a cohesive value type grouping what belongs
  together, not a setter.

### I3 — No usage leak

A class carries nothing that exists for one particular caller. Probe-only, test-only,
training-only or GUI-only needs are solved outside the class: a dedicated collaborator, an
abstraction both sides depend on, a decorator, a subclass in the caller's module, or a free
function over the existing surface.

- Known instance: no training-only signal inside `arenai_model`.
- Known instance: RmlUi types never leave `arenai_desktop/src/gui/`.
- Test: "delete that caller — is this member dead?" Yes ⟹ violation.

### Frozen contracts and no backward compatibility

- `TorchAction`, `TorchState`, `TorchStep` (`agents/torch_types.h`) are shared with code outside
  this repo (Samuel's SAC on another machine): never add a field. Algorithm-specific data
  (pre-tanh sample, log-probs, hidden states) goes in the buffers' `*InputStep` / `*Rollout` and
  the collector's `on_act`.
- A validated feature is permanent: no disable option, no « 0 = off » value, no fallback for old
  `config.json` or checkpoints. New logic gets its own class instead of being duplicated across
  the trainers. Old runs are probed with a binary built in a worktree at their commit.

### I4 — Minimal signatures

No boolean or enum *mode* parameter, no optional parameter, no default argument added to spare
the existing call sites. A new need is a new type or a new responsibility, not a new knob on an
existing signature.

- Smell: `f(..., bool with_x = false)`; `if (mode)` at the top of a method.
- Smell: a default argument whose value is the "old behaviour".
- Two behaviours ⟹ two named functions, or polymorphism, or a strategy injected per I2.

## Judgement criteria (weigh, then justify in one line each when relevant)

- **Single responsibility**: one reason to change per class. A class name containing "And",
  "Manager", "Helper", "Utils" for a stateful thing is a warning.
- **Cohesion**: every member is used by most methods. A member used by one method is an operand
  or belongs to another class.
- **Dependency direction**: `utils → model → view → core → agent / desktop`. Never upward, never
  sideways between `agent` and `desktop`. Inside a module, depend on abstractions at boundaries.
- **Composition over inheritance**: inherit only for a genuine substitutable interface
  (`Abstract*` bases). Never to share code.
- **Immutability**: `const` by default; value types over mutable shared state; no output
  parameters — return a value or a struct.
- **Functional core**: computations (losses, geometry, returns, statistics) are pure functions of
  their arguments; side effects (I/O, optimizer steps, metrics recording) sit at the edge.
- **Ownership is explicit**: `unique_ptr` for sole owner, `shared_ptr` only for real sharing,
  reference/pointer for non-owning use. Do not widen ownership to make something compile.
- **Naming is semantic**: a name states what the thing *is* in the domain, not how it is used
  here. Vocabulary of the codebase: "agent", never "algorithm".
- **Leanest variant**: every abstraction must be justified by a present need. No speculative
  generality, no layer that only forwards.
- **Symmetry**: sibling classes (the agents, the distributions, the metrics) keep parallel
  shapes. A change to one sibling's surface is a question about all of them.

## Decision procedure for any change

1. **Locate the need's owner.** Which class holds the state or the knowledge? The change goes
   there. If no class owns it, the missing thing is a type, not a method on a neighbour.
2. **Classify what crosses**: collaborator (I2 → constructor) or operand (parameter).
3. **Enumerate designs** — at least two real ones for any shared surface: e.g. move behaviour
   into owner / new value type / new collaborator injected / new abstraction / free pure
   function. "Add a getter" and "add a parameter" are candidates only if they pass I1–I4.
4. **Check I1–I4** on each. Discard violators.
5. **Rank the survivors** on the judgement criteria. State the cost: files touched, call sites
   rewritten, siblings to realign.
6. **Gate** (see below). The cheapest diff is never an argument by itself.

## Gate

- Change touches a **shared surface**, existing or newly created ⟹ **STOP before writing code**.
  Ask through `AskUserQuestion`: one question per independent choice, 2–3 concrete designs, the
  recommended one first, each with its exact proposed diff in the option preview — file,
  signature added / changed / removed, who implements it, who calls it — and its cost. A
  one-sentence description (« on expose la matrice ») is not a proposal. Samuel's answer validates that signature only, not the neighbouring choices.
- Change stays inside one class's private part or one `.cpp` and alters neither contract nor
  intended behaviour ⟹ free.
- **Any doubt** about whether a change is right under this doctrine — even on a non-shared
  surface — ⟹ ask. Doubt is the trigger; certainty must be earned by steps 1–5.
- While implementing a validated design, discovering that another shared surface must move ⟹
  stop and ask again. Approval does not propagate.

## Recorded failures (do not repeat)

- Added accessors exposing internals so that a caller could finish the computation (I1).
- Added a parameter to a method for a value the object should have received at construction
  (I2).
