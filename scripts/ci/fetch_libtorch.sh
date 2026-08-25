#!/usr/bin/env bash
# Downloads LibTorch (CPU) for the current architecture into <dest>.
#
# x86_64 has an official libtorch zip; there is none for aarch64, so there the
# lib is extracted from the manylinux pip wheel — its torch/ directory has the
# exact libtorch layout (lib/, include/, share/cmake) and works as
# LIBTORCH_PATH directly.
#
# Usage: fetch_libtorch.sh <version> <dest_dir>
set -euo pipefail

VERSION=${1:?usage: fetch_libtorch.sh <version> <dest_dir>}
DEST=${2:?missing <dest_dir>}

if [ -f "$DEST/share/cmake/Torch/TorchConfig.cmake" ]; then
    echo "libtorch already present at $DEST"
    exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST"

case "$(uname -m)" in
    x86_64)
        curl -fL -o "$TMP/libtorch.zip" \
            "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${VERSION}%2Bcpu.zip"
        unzip -q "$TMP/libtorch.zip" -d "$TMP"
        cp -r "$TMP/libtorch/." "$DEST"
        ;;
    aarch64)
        # cp312 is arbitrary: the C++ payload of the wheel is identical across
        # Python versions, and libtorch_python.so is never linked by ArenAI.
        curl -fL -o "$TMP/torch.whl" \
            "https://download.pytorch.org/whl/cpu/torch-${VERSION}%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl"
        unzip -q "$TMP/torch.whl" -d "$TMP/wheel" "torch/*"
        cp -r "$TMP/wheel/torch/." "$DEST"
        ;;
    *)
        echo "unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

test -f "$DEST/share/cmake/Torch/TorchConfig.cmake"
echo "libtorch ${VERSION} installed at $DEST"
