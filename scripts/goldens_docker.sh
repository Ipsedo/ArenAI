#!/usr/bin/env bash
# Run the golden-image tests in the CI render environment, or regenerate them.
#
#   ./scripts/goldens_docker.sh              # compare, exactly like the CI does
#   ./scripts/goldens_docker.sh --regenerate # rewrite the committed goldens
#
# The goldens record what llvmpipe rasterizes on Ubuntu 24.04, down to the Mesa
# and Bullet versions. Regenerating them from a desktop GPU (or from Arch's much
# newer Mesa) produces references the CI can never match, whatever the tolerance
# — hence this container, which is the same one the CI job runs in.
#
# Everything lands in cmake-build-docker (gitignored, kept out of the host
# build tree which is configured for the host toolchain). LibTorch and
# ccache are cached on the host between runs; the first run downloads ~500 MB.
set -euo pipefail

# Keep in sync with .github/workflows/arenai-ci.yml
IMAGE="${ARENAI_CI_IMAGE:-ubuntu:24.04}"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.12.0}"

REGENERATE=0
case "${1:-}" in
    --regenerate) REGENERATE=1 ;;
    "") ;;
    *)
        echo "usage: $0 [--regenerate]" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/arenai-ci"
mkdir -p "$CACHE_DIR"

# The repo is mounted at its host path, to keep paths in error messages clickable.
exec docker run --rm -i \
    -v "$REPO_ROOT:$REPO_ROOT" \
    -v "$CACHE_DIR:/cache" \
    -w "$REPO_ROOT" \
    -e LIBTORCH_VERSION="$LIBTORCH_VERSION" \
    -e REGENERATE="$REGENERATE" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "$IMAGE" bash -euo pipefail -s <<'CONTAINER'
scripts/ci/install_ubuntu_deps.sh

# LibTorch is only needed for find_package(Torch) to succeed at configure time;
# nothing it provides affects the rendered pixels.
if [ ! -d /cache/libtorch ]; then
    echo "--- downloading libtorch ${LIBTORCH_VERSION} (cached on the host for next runs)"
    curl -fL -o /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
    unzip -q /tmp/libtorch.zip -d /cache
    rm /tmp/libtorch.zip
fi

export CCACHE_DIR=/cache/ccache

cmake -S . -B cmake-build-docker \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBTORCH_PATH=/cache/libtorch \
    -DARENAI_REGENERATE_GOLDEN_IMAGES="$([ "$REGENERATE" -eq 1 ] && echo ON || echo OFF)" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build cmake-build-docker -j"$(nproc)" \
    --target arenai_view_tests arenai_core_tests

. scripts/ci/render_env.sh
ctest --test-dir cmake-build-docker --output-on-failure -R 'arenai_(view|core)_tests'

# docker runs as root; hand the regenerated goldens back to the caller
chown -R "${HOST_UID}:${HOST_GID}" \
    arenai_core/tests/resources/golden_images \
    arenai_view/tests/resources/golden_images \
    cmake-build-docker

if [ "$REGENERATE" -eq 1 ]; then
    echo "--- goldens rewritten: review 'git diff' and commit them"
fi
CONTAINER
