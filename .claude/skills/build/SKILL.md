---
name: build
description: Build the ArenAI C++ project with CMake + make. Use whenever the project must be built, compiled or rebuilt (« build », « compile », « rebuild », « recompile »), including a probe binary at another commit in a worktree.
---

# Build

From the repo root (directory of the top-level `CMakeLists.txt`):

```shell
./build_linux.sh
```

Equivalent to `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3" ..` then
`make -j "$(nproc)"` inside `build/`. Incremental. This is the tree the pre-commit hooks and
`ctest` use.

- **LibTorch** expected in `/opt/libtorch`; elsewhere: configure by hand with
  `-DLIBTORCH_PATH=/path/to/libtorch`.
- `cmake-build-release/` is Samuel's CLion/ninja tree, usually the freshest binaries:
  `cmake --build cmake-build-release --target <t>` builds one target there.
- Probe binary for an older run: `git worktree add <scratch>/<hash> <hash>`, run
  `build_linux.sh` there, pass the resulting `arenai_agent_probe` with `--binary`. Remove the
  worktree afterwards.
- Clean build only when incremental state is proven wrong: `rm -rf build`.
- Windows: `.\build_windows.ps1` (`-Config Debug`, `-LibtorchPath …`).
- Cold build takes several minutes: Bash timeout ≥ 10 min or `run_in_background`.
- Report faithfully: on failure quote the first compiler/linker error; claim success only when
  `make` ended without error.
