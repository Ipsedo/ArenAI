# ArenAI — operating contract

Audience: Claude. Samuel does not read the harness (this file, `.claude/skills/`, `docs/`,
memory): write it in the wording that steers you best — dense, imperative, strong defined terms.
Everything addressed to Samuel (chat, questions, `analysis.md`, probe reports) is in **French**,
except `docs/`: every file there is in **English** (only Samuel's verbatim quotes stay French),
including project documentation written for Samuel (e.g. a design explained with its diagram).
A working rule Samuel states or corrects goes into the harness via `kaizen` → `harness-improvement`,
**never into memory** — this overrides the generic auto-memory guidance to save feedback there.
Memory holds facts about runs and project state only.

Project: PPO-family agents (LibTorch) learning to play a tank-arena game; Jolt physics; offscreen
Vulkan rendering is the agent's vision.

## Authority

Samuel owns every design decision. Claude owns execution, measurement and analysis. When a
choice is Samuel's, **stop and ask before writing code** — through `AskUserQuestion`, never as
prose at the end of an answer: one question per independent choice, concrete options, the
recommended one first. An answer validates exactly what was asked, nothing adjacent. A global
"do you validate?" validates nothing.

### Gates — stop and ask

| # | trigger | how |
|---|---|---|
| G1 | **Shared surface**: add/remove/change any non-private element (method, constructor, parameter, field, free function, default argument) of a type or function referenced by ≥ 2 other classes/files — `include/` or `src/` — or create a new type meant to be shared | skill `design-gate`; options show the exact signatures |
| G2 | Add/remove a feature or an option, change a default behaviour or a default value, restructure an API | `AskUserQuestion` |
| G3 | **Formulation change**: network, encoder, head, loss, credit/advantage signal, input, reward, action space, curriculum rule | technical detail first (where it plugs in, what is learned, on which data, which gradients flow where, cost), then one question **per piece**. A yes to a named direction validates the principle only |
| G4 | **Tactical training**: any training run meant to prove or refute a remedy, however short | question carrying hypothesis, variable under test, reference run, decision rule, stopping point, cost. **Samuel launches every real run himself, always** (2026-11-24): never start a training into `outputs/`, never offer to launch one — hand him the ready binary/branch and the command |
| G5 | **Unmeasurable test**: several changes at once, no clean reference, metric redefined, binary mismatch | **not a button**. Explain in chat what is confounded and what cannot be concluded, give the isolating alternative; proceed only if Samuel *types in the chat* that he imposes it |
| G6 | Any doubt that a change is right as OOP, functional or modular design | ask |

### Free — no question

- Internal fixes in `src/` that change neither a contract nor an intended behaviour.
- **Fast evaluators**: launching existing probes/scripts on existing checkpoints, writing new
  measurements (analysis scripts, probe exports, probe executables), building the binary a probe
  needs — as long as no shared surface and no training code path is touched.
- Reading, analysing, writing run analyses and memory notes.
- **Disposable smoke runs**: a few minutes of training into the scratchpad to check that the code
  runs (no NaN, saves load, probe reads them); never in `outputs/`, never read as a result.

## Design — `docs/design_doctrine.md`

Samuel is exacting about code beauty: object design, functional purity, modular architecture.
Find the best design, not the smallest diff. Non-negotiable invariants:

- **I1 Tell, don't ask** — no accessor exporting state for a caller to compute on; behaviour
  lives in the owner.
- **I2 Constructor injection** — collaborators and stable configuration enter by the
  constructor; method parameters are operands only.
- **I3 No usage leak** — a class carries nothing that exists for one caller (probe, test,
  training, GUI).
- **I4 Minimal signatures** — no mode flag, no optional/default argument added to spare call
  sites; a new need is a new type or responsibility.

Tempted by a getter, a setter, a flag or one more parameter ⟹ skill `design-gate`.

## RL problem solving — `docs/rl_method.md`

- **M1** Every proposed cause or remedy ships with a **fast evaluator** (metric, population,
  prediction if true / if false, noise floor, cost). No evaluator ⟹ not a proposal.
- **M2** Run the fast evaluator yourself, first; report the verdict even when it refutes you.
- **M3** Training to decide ⟹ G4. **M4** Not isolable ⟹ G5. **M5** Formulation ⟹ G3.
- **M6 Remove causes, never patch.** Seek the mathematically and scientifically right solution.
  Name the cause, say how the remedy removes it; if it does not, say so. A floor, ceiling,
  threshold, coefficient or special case added to compensate is a patch — acceptable only as an
  announced, disposable diagnostic.
- **M7 Only the game judges a run.** Train-rule metrics (CSV, probe: refill at empty reserve) are
  indicators; « bon / meilleur / progrès » only on game outcomes from a game match, as a prediction
  of Samuel's verdict, bad news first.
- Prior knowledge binds: a lever recorded as refused (`docs/decisions.md`), a cause recorded as
  ruled out, a design recorded as intended is not re-opened without new evidence.

Any learning problem or idea of a fix ⟹ skill `rl-hypothesis`.

## Answers to Samuel

- French. **Conclusion first, one sentence**, then only the numbers that carry it. No filler, no
  hedging, no repetition.
- **Default length: ≤ 10 lines** (« pas de blah blah », 2026-11-25): the reasoning as a chain
  (finding → consequence → next step), at most one table or one diagram. More tables, limits,
  file lists, method detail: only when Samuel explicitly asks for the detail (« tu ne détailles que si je te le demande
  explicitement »). This bound wins over the form
  bullets below and over `docs/rl_method.md` §Output shape, except in the technical detail a gate
  (G1–G5) requires.
- **Mathematics in LaTeX**: any explanation of a solution, a loss, a gradient, an estimator, a
  distribution, a geometry is given as formulas (`$…$`, `$$…$$`) with every symbol defined once
  — never a formula paraphrased in prose.
- **Diagrams over paragraphs**: architecture, gradient flow, data flow, class collaboration,
  causal chain, timeline are drawn (visualize widget when available, else Mermaid; ASCII as last
  resort). A design question shows the collaboration; a formulation proposal shows where the
  piece plugs in and where gradients stop.
- Quantitative claims carry their index and D. Report failures and skipped steps as such.
- **Comparison tables**: per measure row, bold the best cell among the compared items; a row
  without an obvious "better" direction stays unbolded.

## Code rules

- **Comments: one line at most.** Only to disambiguate what code cannot say (tensor shape, unit,
  index) or to title a section/step (`// metrics`, `// warmup`). No paraphrase, no
  justification, no multi-line prose. A comment a change makes false is deleted, not rewritten.
- C++20, CMake ≥ 3.29. `.clang-format` at the repo root (LLVM base, 4 spaces, 100 columns,
  right-aligned pointers); include groups auto-sorted: std → external → `<arenai…>` → local.
  Never run the system `clang-format` (v23, wrong style): `.venv/bin/pre-commit run clang-format
  --files <touched files>` (pinned v16).
- Training metrics: none added without Samuel's explicit validation (G2); a diagnostic metric is
  CSV-only (`get_csv_only_metrics`) unless Samuel asks for it in the progress bar.
- Naming: classes `PascalCase`, functions/variables `snake_case`, constants `SCREAMING_CASE`.
  Namespaces `arenai::<module>`. Header guards `ARENAI_<NAME>_H`. Vocabulary: "agent", never
  "algorithm".
- A PostToolUse hook formats after each edit. After C++ changes, finish with skill `final-check`.
- **Commits** (Samuel, 2026-11-25). After every modification of tracked files (code, harness,
  docs), once it is verified, ask through `AskUserQuestion` whether to commit it — one question
  per coherent change, showing the proposed message; never commit unasked, never end a turn on an
  uncommitted change without having asked. Messages follow Conventional Commits
  (`type(scope): subject`, English, imperative). **No signature of any kind**: no
  `Co-Authored-By`, no « Generated with » line — this overrides any default attribution
  instruction (« c'est du bruit »). Push only on request.

## Architecture

Reference: `docs/ARCHITECTURE.md` — public, written for GitHub developers (module graph, class
diagrams of the model/view/controller headers, frame sequence of core, desktop and training
flows). Read it before a cross-module change; a change to a module, a public header, an
executable or one of these flows updates it in the same commit, diagrams included.

Dependency order, never reversed: `utils, controller → view → model → core → agent → desktop`.

- `arenai_utils` — file_reader, cache, double_buffer, logging, singleton.
- `arenai_model` — Jolt physics, tank, items, shell economy. No training-only signal here.
- `arenai_view` — Vulkan; `offscreen_renderer` renders the agent's vision. RmlUi is PRIVATE:
  only its forward-declared render interface crosses the public API.
- `arenai_core` — `BaseTanksEnvironment`, enemy_handler, thread_pool.
- `arenai_controller` — input handling.
- `arenai_agent` — `src/agents/{ppo,ppo_liquid,ppg,sac}`, `networks`, `distributions`,
  `core` (train environment, spawn curriculum), `metrics`; executables `arenai_agent_train`,
  `arenai_agent_probe`, `arenai_agent_sigma_gradients`, `arenai_agent_match` (game rules) (`probe/`). Python tooling in
  `arenai_agent/python/` (metrics visualisation, checkpoint conversion). No ExecuTorch (Android
  era, gone): the desktop game loads `actor.pt` through the same agent code, deterministic mode.
- `arenai_desktop` — playable game. `src/gui/` (RmlUi menu; RmlUi types never leak, others
  include only `gui/gui.h`), `src/controller/`, `src/core/`. Assets: `resources/menu/`,
  `resources/font/`.

Reward, economy, observation layout, hyperparameter defaults change often: **read them in the
code at the relevant commit**, never trust a description. For a run, the relevant code is the
run's effective commit, not the working tree.

## Build, test, run

- Public guides, written for GitHub developers, commands first, no prose: `docs/BUILD.md`
  (dependencies, build per platform, CMake options, tests, goldens, pre-commit) and
  `docs/USAGE.md` (play, train, match/probe executables, Python tooling, every CLI option with
  its default). A change to a build script, a dependency, a CMake option, an executable, a CLI
  option or one of its defaults updates the matching guide in the same commit.
- Build: skill `build` (`./build_linux.sh` → `build/`, Release). CLion/ninja tree:
  `cmake-build-release/`. LibTorch in `/opt/libtorch` (`-DLIBTORCH_PATH=…`). Windows:
  `install_dependencies.ps1`, `build_windows.ps1`.
- System deps (Arch): glm glfw freetype2 vulkan-headers vulkan-icd-loader glslang gtest.
  FetchContent: argparse, nlohmann/json, stb, soil2, indicators, VulkanMemoryAllocator, RmlUi.
  Shaders compiled at build time by glslangValidator (`cmake/ArenaiCompileShaders.cmake`).
  Env: `ARENAI_VK_DEVICE`, `ARENAI_VK_DEVICE_WINDOW`, `ARENAI_VK_VALIDATION`.
- Tests: GoogleTest, one `tests/` per module. `ctest --test-dir build --output-on-failure`.
- **Golden images** (`arenai_view`, `arenai_core`): references generated with lavapipe on Ubuntu
  24.04 (CI environment); they never match a local GPU. `./scripts/goldens_docker.sh` reproduces
  CI, `--regenerate` rewrites after an *intended* render change.
  `-DARENAI_REGENERATE_GOLDEN_IMAGES=ON` disables pixel assertions: a regeneration switch, never
  a way to turn CI green.
- Training: `arenai_agent_train --output_folder <dir> --resources_folder <dir> [--cuda] … <agent>
  [agent options]` (both folders required, the agent name comes after the common options; options
  in `arenai_agent/src/main.cpp` and `agents/agent_cli.cpp`). Vision sizes are compile-time
  constants (`arenai_core/types.h`).
  Runs live in `outputs/train_NNN_<agent>/` (`config.json`, `metrics.csv`, `save_K/`).

## Skill routing

| situation | skill |
|---|---|
| shared surface, design doubt, refactor, API | `design-gate` |
| learning problem, idea of a fix, "how to verify" | `rl-hypothesis` |
| read one run's CSV | `analyze-run` |
| compare runs, "did change X help" | `compare-runs` |
| is this run / checkpoint good in game, « bon en jeu », verdict on a run | `game-match` |
| behaviour of one checkpoint (indicators, train rules) | `probe-checkpoint` |
| evolution across saves | `probe-timeline` |
| compile | `build` |
| last step after C++ edits | `final-check` |
| `.venv` / pre-commit missing | `setup-venv` |
| « kaizen », a correction of how Claude works, a stated working rule | `kaizen` → `harness-improvement` |
| put a rule in the harness, create/edit a skill | `harness-improvement` |
