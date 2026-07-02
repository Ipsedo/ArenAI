# ArenAI — C++ core

A neural network (SAC) trained to play a tank-arena game.
Bullet physics, offscreen OpenGL ES 3.0 / EGL rendering (the agent's vision), NN via LibTorch.
Part of the parent **PhyVR** project (Android/VR, Gradle) which reuses this C++ core.

## Architecture (CMake modules)

Dependency order: utils → model → view → core → train / desktop

- **arenai_utils**  — helpers (file_reader, cache, double_buffer, logging, singleton)
- **arenai_model**  — physics & entities: Bullet engine, tank, items (`arenai_model/`)
- **arenai_view**   — OpenGL rendering; `pbuffer_renderer` = offscreen render for the agent's vision
- **arenai_core**   — `BaseTanksEnvironment`, enemy_handler, thread_pool (RL env loop)
- **arenai_controller** — input handling
- **arenai_train**  — SAC: agents, networks, replay_buffer, reward_transforms + `main.cpp` (training executable)
- **arenai_desktop** — the playable game executable

## Build

C++20, CMake ≥ 3.29.

### Linux (ArchLinux)
```shell
sudo pacman -Sy bullet glm glfw
# LibTorch expected in /opt/libtorch (override with -DLIBTORCH_PATH=...)
mkdir build && cd build && cmake .. && make
```

### Windows
```powershell
.\install_dependencies.ps1   # vcpkg + libtorch into .\libs
.\build_windows.ps1          # -Config Debug | -LibtorchPath ... to override
```

Deps fetched via FetchContent (no manual install): argparse, nlohmann/json, stb, soil2, indicators.
System libs: bullet, glm, glfw. LibTorch via `find_package(Torch)`.

## Tests

Framework: **GoogleTest**. One `tests/` folder per module (core, model, view, train).

```shell
cd build && ctest --output-on-failure
# or run a specific binary: ./arenai_train/tests/arenai_train_tests
```

## Running training

```shell
./arenai_train_exec --output_folder <dir> --asset_folder <dir> [--cuda] ...
```
`--output_folder` and `--asset_folder` are **required**. Outputs go to `arenai_train/outputs/train_NNN/`.
Many hyperparameters (tau, gamma, learning rates, nb_tanks, vision_height...) — see `arenai_train/src/main.cpp`.

## Code conventions

Enforced by `.clang-format` and pre-commit — both live in the **parent** `PhyVR/` repo, not in `arenai/`.

- **`.clang-format`** (`../.clang-format`): LLVM base, 4-space indent, 100-col limit, right-aligned pointers.
  Run `clang-format -i` before committing, or install the hooks: `pre-commit install` (from `PhyVR/`).
- Include groups are auto-sorted (`IncludeBlocks: Regroup`): std `<...>` → external `<.../...>` → `<arenai...>` → local `"..."`.
- Naming: classes `PascalCase`, functions/variables `snake_case`, constants `SCREAMING_CASE`.
- No top-level namespace; header guards `ARENAI_<NAME>_H`.

## Python tooling

`arenai_train/python/` (`.venv`): ExecuTorch model conversion + metrics visualization.
