#!/usr/bin/env bash
# Assembles the self-contained Linux release archive:
#
#   arenai-<version>_linux-<arch>.zip
#   ├── arenai_desktop        (looks for ./resources next to itself)
#   ├── arenai_agent_train
#   ├── resources/
#   └── lib/                  (every bundled .so, found via $ORIGIN RPATH)
#
# The executables must have been built with -DARENAI_PORTABLE_RPATH=ON so the
# relocated binaries resolve lib/ wherever the zip is extracted.
#
# Usage: package_linux.sh <build_dir> <version> [out_dir]
set -euo pipefail

BUILD_DIR=$(realpath "${1:?usage: package_linux.sh <build_dir> <version> [out_dir]}")
VERSION=${2:?missing <version>}
OUT_DIR=$(realpath -m "${3:-dist}")

REPO_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")

case "$(uname -m)" in
    x86_64) ARCH=x86_64 ;;
    aarch64) ARCH=arm64 ;;
    *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

ZIP_NAME="arenai-${VERSION}_linux-${ARCH}.zip"

# glibc and the windowing/session stack must come from the user's system: the
# libc family is tied to the dynamic loader, and the X11/Wayland client
# libraries have to match the running session. Mesa's libGL/libEGL belong to
# the GPU driver. Everything else is bundled — including libstdc++/libgcc,
# which are newer than what any target distro needs and backward compatible.
SYSTEM_LIBS='ld-linux|libc\.so|libm\.so|libdl\.so|libpthread\.so|librt\.so'
SYSTEM_LIBS+='|libresolv\.so|libnsl\.so|libutil\.so'
SYSTEM_LIBS+='|libX11|libxcb|libXau|libXdmcp|libXext|libXrandr|libXi\.so|libXcursor'
SYSTEM_LIBS+='|libXinerama|libXrender|libXfixes|libXxf86vm|libxkbcommon|libwayland'
SYSTEM_LIBS+='|libGL\.so|libGLX|libEGL|libGLdispatch|libOpenGL'

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/lib" "$OUT_DIR"

cp "$BUILD_DIR/arenai_desktop/arenai_desktop" "$STAGE/"
cp "$BUILD_DIR/arenai_agent/arenai_agent_train" "$STAGE/"
cp -r "$REPO_DIR/resources" "$STAGE/resources"

# ldd prints the full transitive closure of each executable; keep everything
# that is not a system-provided library.
for exe in arenai_desktop arenai_agent_train; do
    ldd "$STAGE/$exe" | awk '/=>/ { print $3 }' | while read -r lib; do
        [ -f "$lib" ] || continue
        basename=$(basename "$lib")
        if echo "$basename" | grep -qE "$SYSTEM_LIBS"; then
            continue
        fi
        if [ ! -e "$STAGE/lib/$basename" ]; then
            cp "$lib" "$STAGE/lib/"
        fi
    done
done

# Fail loudly if a dependency could not be resolved at packaging time: it
# would be missing on the user's machine too.
for exe in arenai_desktop arenai_agent_train; do
    if ldd "$STAGE/$exe" | grep -q "not found"; then
        echo "unresolved dependencies for $exe:" >&2
        ldd "$STAGE/$exe" | grep "not found" >&2
        exit 1
    fi
done

echo "bundled libraries ($(ls "$STAGE/lib" | wc -l)):"
ls -lh "$STAGE/lib"

(cd "$STAGE" && zip -qr "$OUT_DIR/$ZIP_NAME" .)
echo "wrote $OUT_DIR/$ZIP_NAME ($(du -h "$OUT_DIR/$ZIP_NAME" | cut -f1))"
