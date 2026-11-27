# Build

C++20 · CMake ≥ 3.29 · GCC ≥ 13 or MSVC (VS 2022) · LibTorch 2.12 · Vulkan ≥ 1.2

```bash
git clone https://github.com/Ipsedo/ArenAI.git
cd ArenAI
```

## Dependencies

### Arch Linux

```bash
sudo pacman -S base-devel cmake git glm glfw freetype2 vulkan-headers vulkan-icd-loader glslang gtest
```

### Ubuntu 24.04

```bash
sudo apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git unzip python3 \
    libglm-dev libglfw3-dev libgtest-dev libvulkan-dev \
    mesa-vulkan-drivers glslang-tools libfreetype-dev
```

Ubuntu 24.04 ships CMake 3.28:

```bash
curl -fL https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz |
    sudo tar -xz --strip-components=1 -C /usr/local
```

### LibTorch (Linux)

Default location: `/opt/libtorch`. CPU build, x86_64 or aarch64:

```bash
sudo scripts/ci/fetch_libtorch.sh 2.12.0 /opt/libtorch
```

CUDA build:

```bash
wget https://download.pytorch.org/libtorch/cu132/libtorch-shared-with-deps-2.12.0%2Bcu132.zip
sudo unzip libtorch-shared-with-deps-2.12.0+cu132.zip -d /opt
```

Fetched by CMake at configure time (nothing to install): Jolt Physics, RmlUi, VulkanMemoryAllocator,
argparse, nlohmann/json, stb, SOIL2, indicators.

## Linux

```bash
./build_linux.sh
```

Raw equivalent:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3"
cmake --build build -j "$(nproc)"
```

LibTorch elsewhere:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIBTORCH_PATH=/path/to/libtorch
```

One target:

```bash
cmake --build build -j "$(nproc)" --target arenai_desktop
```

Ninja tree (CLion layout):

```bash
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target arenai_agent_train
```

Clean rebuild:

```bash
rm -rf build && ./build_linux.sh
```

| Target | Output |
|---|---|
| `arenai_desktop` | `build/arenai_desktop/arenai_desktop` |
| `arenai_agent_train` | `build/arenai_agent/arenai_agent_train` |
| `arenai_agent_probe` | `build/arenai_agent/arenai_agent_probe` |
| `arenai_agent_match` | `build/arenai_agent/arenai_agent_match` |
| `arenai_agent_sigma_gradients` | `build/arenai_agent/arenai_agent_sigma_gradients` |
| `arenai_agent_render_bench` | `build/arenai_agent/arenai_agent_render_bench` |
| `arenai_{view,model,core,agent,desktop}_tests` | `build/arenai_<module>/tests/` |

## Windows

Requires Git and Visual Studio 2022 with the "Desktop development with C++" workload.

```powershell
.\install_dependencies.ps1
.\build_windows.ps1
```

Binaries and DLLs land in `build\bin\`. Dependencies (vcpkg, LibTorch) land in `libs\`.

| Script | Parameter | Default | Effect |
|---|---|---|---|
| `install_dependencies.ps1` | `-Cuda` | off | CUDA LibTorch instead of CPU |
| `install_dependencies.ps1` | `-SoftwareVulkan` | off | Also fetch Mesa lavapipe (no-GPU fallback) |
| `install_dependencies.ps1` | `-LibtorchPath <dir>` | `libs\libtorch` | LibTorch destination |
| `install_dependencies.ps1` | `-VcpkgRoot <dir>` | `libs\vcpkg` | vcpkg location |
| `build_windows.ps1` | `-Config <cfg>` | `Release` | Build type (`Debug`, `Release`) |
| `build_windows.ps1` | `-LibtorchPath <dir>` | `libs\libtorch` | LibTorch location |
| `build_windows.ps1` | `-VcpkgRoot <dir>` | `libs\vcpkg` | vcpkg location |

## macOS (arm64, experimental)

CI packaging recipe only; not a maintained local build. Prebuilt `ArenAI.app`:
[releases](https://github.com/Ipsedo/ArenAI/releases).

```bash
brew install cmake glfw glm freetype glslang googletest vulkan-headers vulkan-loader molten-vk
scripts/ci/fetch_libtorch.sh 2.12.0 "$HOME/libtorch"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIBTORCH_PATH="$HOME/libtorch"
cmake --build build -j "$(sysctl -n hw.ncpu)" --target arenai_desktop arenai_agent_train
```

## CMake options

| Option | Default | Effect |
|---|---|---|
| `LIBTORCH_PATH` | `/opt/libtorch` | LibTorch root directory |
| `CMAKE_BUILD_TYPE` | unset | `Release` for any real use |
| `ARENAI_PORTABLE_RPATH` | `OFF` | Relocatable RPATH for packaged releases |
| `ARENAI_REGENERATE_GOLDEN_IMAGES` | `OFF` | Tests rewrite goldens, no pixel assertion |
| `MESA_PATH` | `libs/mesa/x64` | Windows: Mesa DLLs copied next to binaries |
| `CMAKE_OSX_DEPLOYMENT_TARGET` | `13.3` | Minimum macOS version |

## Tests

```bash
ctest --test-dir build --output-on-failure
```

One module:

```bash
ctest --test-dir build --output-on-failure -R arenai_model_tests
```

### Golden images

References are rendered by lavapipe on Ubuntu 24.04 (the CI container); they never match a local
GPU, so pixel comparisons are skipped outside that environment. Requires Docker. Goldens outside
`arenai_view` render a simulated scene: they also encode physics poses, so a change of physics
behaviour breaks them like a render change.

```bash
./scripts/goldens_docker.sh                # compare, as CI does
./scripts/goldens_docker.sh --regenerate   # rewrite after an intended render change
```

| Variable | Default | Effect |
|---|---|---|
| `ARENAI_CI_IMAGE` | `ubuntu:24.04` | Container image |
| `LIBTORCH_VERSION` | `2.12.0` | LibTorch downloaded into the cache |

Build tree: `cmake-build-docker/`. Cache: `${XDG_CACHE_HOME:-~/.cache}/arenai-ci`.

## Runtime environment variables

| Variable | Value | Effect |
|---|---|---|
| `ARENAI_VK_DEVICE` | index or device-name substring | GPU of the offscreen renderer (agent vision); default: integrated first |
| `ARENAI_VK_DEVICE_WINDOW` | index or device-name substring | GPU of the game window; default: discrete first |
| `ARENAI_VK_VALIDATION` | any value | Enable `VK_LAYER_KHRONOS_validation` when installed |

```bash
ARENAI_VK_DEVICE=0 ARENAI_VK_DEVICE_WINDOW=NVIDIA ARENAI_VK_VALIDATION=1 <executable> …
```

## Formatting and pre-commit

```bash
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip pre-commit
./.venv/bin/pre-commit install
```

Format touched files (pinned clang-format 16; do not use the system one):

```bash
git add <files>
./.venv/bin/pre-commit run clang-format --files <files>
```

Full check — format, `./build_linux.sh`, `ctest`:

```bash
./.venv/bin/pre-commit run --all-files
```

---

Running the game, training and probes: [USAGE.md](USAGE.md). Module layout: [ARCHITECTURE.md](ARCHITECTURE.md).
