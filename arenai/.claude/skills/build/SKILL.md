---
name: build
description: Build the ArenAI C++ project with CMake + make. Use whenever the user wants to build, compile, or rebuild the project (e.g. "build it", "compile", "run a build", "rebuild").
---

# Build

Configure and compile the project from the repo root, parallelizing over all available CPU cores.

## Steps

Run from the **project root** (`arenai/`, the directory containing the top-level `CMakeLists.txt`):

```shell
mkdir -p build && cd build
cmake .. && make -j "$(nproc)"
```

- `mkdir -p build` — create the build directory if it doesn't already exist (safe to re-run; won't fail on an existing build).
- `-j "$(nproc)"` — parallel build using the number of available CPU cores. On macOS use `-j "$(sysctl -n hw.ncpu)"`; on Windows/MSVC prefer the PowerShell scripts (see below).

## Notes

- **LibTorch** is expected in `/opt/libtorch`. If it lives elsewhere, pass `-DLIBTORCH_PATH=/path/to/libtorch` to the `cmake ..` step.
- Re-running is incremental — CMake reconfigures only if needed and `make` rebuilds only changed targets. To force a clean build, remove the directory first: `rm -rf build`.
- Report the outcome honestly: if the build fails, surface the compiler/linker error; only state success when `make` actually completed without errors.
- On **Windows**, build via `.\build_windows.ps1` (`-Config Debug` / `-LibtorchPath ...` to override) instead of the commands above.
