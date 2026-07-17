#!/usr/bin/env bash
# Installs the build + render toolchain on Ubuntu 24.04.
#
# Shared by the CI container (.github/workflows/arenai-ci.yml) and by
# scripts/goldens_docker.sh so that both get the exact same Mesa: the golden
# images record llvmpipe's rasterization, so any renderer difference between
# the two shows up as failing pixel comparisons.
set -euo pipefail

# Ubuntu 24.04 ships CMake 3.28, the project needs >= 3.29
CMAKE_VERSION="${CMAKE_VERSION:-3.31.6}"

if [ "$(id -u)" -ne 0 ]; then
    echo "$0: must run as root, inside the Ubuntu container" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    curl \
    git \
    unzip \
    libbullet-dev \
    libglm-dev \
    libglfw3-dev \
    libgtest-dev \
    libegl-dev \
    libgles-dev \
    libgl1-mesa-dri \
    libfreetype-dev

if ! cmake --version 2>/dev/null | head -1 | grep -q "${CMAKE_VERSION}"; then
    curl -fL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" |
        tar -xz --strip-components=1 -C /usr/local
fi

# The goldens are tied to this version: a Mesa bump is the expected reason for
# the pixel comparisons to start failing (see scripts/goldens_docker.sh).
echo "mesa (libgl1-mesa-dri): $(dpkg-query -W -f='${Version}' libgl1-mesa-dri)"
echo "cmake: $(cmake --version | head -1)"
