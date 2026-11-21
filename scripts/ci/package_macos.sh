#!/usr/bin/env bash
# Usage: package_macos.sh <build_dir> <version> <libtorch_dir> [out_dir]
set -euo pipefail

BUILD_DIR=$(cd "${1:?usage: package_macos.sh <build_dir> <version> <libtorch_dir> [out_dir]}" && pwd)
VERSION=${2:?missing <version>}
LIBTORCH_DIR=$(cd "${3:?missing <libtorch_dir>}" && pwd)
OUT_DIR=${4:-dist}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

case "$(uname -m)" in
    arm64) ARCH=arm64 ;;
    x86_64) ARCH=x86_64 ;;
    *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

ZIP_NAME="arenai-${VERSION}_macos-${ARCH}.zip"
EXECUTABLES=(arenai_desktop arenai_agent_train)

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

CONTENTS="$STAGE/ArenAI.app/Contents"
MACOS="$CONTENTS/MacOS"
FRAMEWORKS="$CONTENTS/Frameworks"
ICD_DIR="$CONTENTS/Resources/vulkan/icd.d"
mkdir -p "$MACOS" "$FRAMEWORKS" "$ICD_DIR"

# bundle layout
cp "$BUILD_DIR/arenai_desktop/arenai_desktop" "$MACOS/"
cp "$BUILD_DIR/arenai_agent/arenai_agent_train" "$MACOS/"
cp -R "$REPO_DIR/resources" "$MACOS/resources"

cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>ArenAI</string>
    <key>CFBundleIdentifier</key><string>io.github.ipsedo.arenai</string>
    <key>CFBundleExecutable</key><string>arenai_desktop</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundleVersion</key><string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key><string>$(sw_vers -productVersion | cut -d. -f1).0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

# dylib closure into Contents/Frameworks
SEARCH_ARGS=(-s "$LIBTORCH_DIR/lib")
while IFS= read -r dir; do SEARCH_ARGS+=(-s "$dir"); done \
    < <(find "$BUILD_DIR" -name "*.dylib" -exec dirname {} \; | sort -u)

BUNDLER_ARGS=()
for exe in "${EXECUTABLES[@]}"; do BUNDLER_ARGS+=(-x "$MACOS/$exe"); done

dylibbundler -of -b "${BUNDLER_ARGS[@]}" -d "$FRAMEWORKS" \
    -p "@executable_path/../Frameworks/" "${SEARCH_ARGS[@]}" < /dev/null

# Vulkan driver: loaded at runtime by the loader, never linked
MOLTENVK_PREFIX=$(brew --prefix molten-vk)
cp "$(find -L "$MOLTENVK_PREFIX" -name libMoltenVK.dylib | head -1)" "$FRAMEWORKS/"
python3 - "$(find -L "$MOLTENVK_PREFIX" -name MoltenVK_icd.json | head -1)" \
    "$ICD_DIR/MoltenVK_icd.json" <<'EOF'
import json
import sys

with open(sys.argv[1]) as source:
    manifest = json.load(source)
manifest["ICD"]["library_path"] = "../../../Frameworks/libMoltenVK.dylib"
with open(sys.argv[2], "w") as target:
    json.dump(manifest, target, indent=4)
EOF

if [ ! -f "$FRAMEWORKS/libvulkan.1.dylib" ]; then
    echo "libvulkan.1.dylib was not bundled" >&2
    exit 1
fi

# no dependency may point outside the bundle or the system
find "$MACOS" "$FRAMEWORKS" -type f \( -perm -u+x -o -name "*.dylib" \) | while read -r binary; do
    file "$binary" | grep -q "Mach-O" || continue
    if otool -L "$binary" | tail -n +2 | awk '{ print $1 }' \
        | grep -vE '^(@executable_path|@loader_path|@rpath|/usr/lib/|/System/)'; then
        echo "$binary still depends on the paths above" >&2
        exit 1
    fi
done

# arm64 refuses binaries whose signature was invalidated by the path rewrites
find "$MACOS" "$FRAMEWORKS" -type f \( -perm -u+x -o -name "*.dylib" \) | while read -r binary; do
    file "$binary" | grep -q "Mach-O" || continue
    codesign --force --sign - "$binary"
done

echo "bundled libraries ($(ls "$FRAMEWORKS" | wc -l | tr -d ' ')):"
ls -lh "$FRAMEWORKS"

(cd "$STAGE" && zip -qry "$OUT_DIR/$ZIP_NAME" ArenAI.app)
echo "wrote $OUT_DIR/$ZIP_NAME ($(du -h "$OUT_DIR/$ZIP_NAME" | cut -f1))"
