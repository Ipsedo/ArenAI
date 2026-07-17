# ArenAI — C++ core

## Règle de travail — validation obligatoire

**Aucun choix de conception sans validation explicite de Samuel.** Ne jamais, de ta propre
initiative : modifier un contrat d'interface public (`*/include/`), ajouter/retirer une
fonctionnalité ou une option, changer un comportement par défaut, ou restructurer une API.
En cas de doute ou d'alternative de design : **poser la question d'abord**, proposer les
options, attendre la réponse. Les corrections de bugs internes (fichiers `src/`) qui ne
changent ni contrat ni comportement voulu restent autorisées.

A neural network (SAC) trained to play a tank-arena game.
Bullet physics, offscreen Vulkan rendering (the agent's vision), NN via LibTorch.

## Architecture (CMake modules)

Dependency order: utils → model → view → core → train / desktop

- **arenai_utils**  — helpers (file_reader, cache, double_buffer, logging, singleton)
- **arenai_model**  — physics & entities: Bullet engine, tank, items (`arenai_model/`)
- **arenai_view**   — Vulkan rendering; `offscreen_renderer` = offscreen render for the agent's vision
- **arenai_core**   — `BaseTanksEnvironment`, enemy_handler, thread_pool (RL env loop)
- **arenai_controller** — input handling
- **arenai_train**  — SAC: agents, networks, replay_buffer, reward_transforms + `main.cpp` (training executable)
- **arenai_desktop** — the playable game executable. Its `src/` folders are hexagons of their own:
  `gui/` (RmlUi main menu — RmlUi types must never leak out of it, other code only includes
  `gui/menu.h`), `controller/`, `core/`. Menu assets live in `resources/menu/` + `resources/font/`.

## Build

C++20, CMake ≥ 3.29.

### Linux (ArchLinux)
```shell
sudo pacman -Sy bullet glm glfw freetype2 vulkan-headers vulkan-icd-loader
# LibTorch expected in /opt/libtorch (override with -DLIBTORCH_PATH=...)
mkdir build && cd build && cmake .. && make
```

### Windows
```powershell
.\install_dependencies.ps1   # vcpkg (incl. vulkan-headers/loader) + libtorch into .\libs
.\build_windows.ps1          # -Config Debug | -LibtorchPath ... to override
```

Deps fetched via FetchContent (no manual install): argparse, nlohmann/json, stb, soil2, indicators,
glslang (build-time GLSL→SPIR-V compiler, see cmake/ArenaiCompileShaders.cmake),
VulkanMemoryAllocator,
RmlUi (PRIVATE in arenai_view — only its forward-declared render interface crosses the public API).
System libs: bullet, glm, glfw, freetype (RmlUi font engine), Vulkan loader (the runtime driver
comes from the GPU's ICD; `ARENAI_VK_DEVICE`/`ARENAI_VK_DEVICE_WINDOW` override the device pick,
`ARENAI_VK_VALIDATION` enables the validation layer when installed). LibTorch via `find_package(Torch)`.

## Tests

Framework: **GoogleTest**. One `tests/` folder per module (core, model, view, train).

```shell
cd build && ctest --output-on-failure
# or run a specific binary: ./arenai_train/tests/arenai_train_tests
```

### Golden images

`arenai_view` and `arenai_core` compare rendered frames against committed `.json`
references (`*/tests/resources/golden_images/`). Those references are **tied to the
renderer**: they are generated with lavapipe (Mesa's software Vulkan) on Ubuntu 24.04,
the environment the CI runs in, so they will not match a local GPU or Arch's Mesa.

```shell
./scripts/goldens_docker.sh              # reproduce the CI comparison locally (docker)
./scripts/goldens_docker.sh --regenerate # rewrite the goldens after an intended render change
```

Regenerate + commit whenever the rendering legitimately changes (shaders, shadows,
post-processing). `-DARENAI_REGENERATE_GOLDEN_IMAGES=ON` disables every pixel
assertion — it is a regeneration switch, never a way to turn a red CI green.

## Running training

```shell
./arenai_train_exec --output_folder <dir> --asset_folder <dir> [--cuda] ...
```
`--output_folder` and `--asset_folder` are **required**. Outputs go to `arenai_train/outputs/train_NNN/`.
Many hyperparameters (tau, gamma, learning rates, nb_tanks, vision_height...) — see `arenai_train/src/main.cpp`.

## Code conventions

Enforced by `.clang-format` and pre-commit — both live in the **parent** `ArenAI/` repo, not in `arenai/`.

- **`.clang-format`** (`../.clang-format`): LLVM base, 4-space indent, 100-col limit, right-aligned pointers.
  Run `clang-format -i` before committing, or install the hooks: `pre-commit install` (from `ArenAI/`).
- Include groups are auto-sorted (`IncludeBlocks: Regroup`): std `<...>` → external `<.../...>` → `<arenai...>` → local `"..."`.
- Naming: classes `PascalCase`, functions/variables `snake_case`, constants `SCREAMING_CASE`.
- No top-level namespace; header guards `ARENAI_<NAME>_H`.

## Python tooling

`arenai_train/python/` (`.venv`): ExecuTorch model conversion + metrics visualization.
