#!/usr/bin/env bash
# Installs the build toolchain for the CD packaging jobs, inside an
# ubuntu:22.04 container (x86_64 and arm64).
#
# 22.04 rather than 24.04 on purpose: the released binaries inherit the glibc
# they were linked against (2.35), so they run on any distro from 2022 on.
# Everything above glibc is bundled into the zip by package_linux.sh.
# The project needs gcc >= 13 (std::format), which 22.04 only has through the
# ubuntu-toolchain-r PPA; the matching newer libstdc++ is bundled too.
#
# Unlike install_ubuntu_deps.sh (CI), no Mesa/render stack: these jobs only
# build and package, tests already ran in the CI job.
set -euo pipefail

CMAKE_VERSION="${CMAKE_VERSION:-3.31.6}"

if [ "$(id -u)" -ne 0 ]; then
    echo "$0: must run as root, inside the Ubuntu container" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    build-essential \
    ca-certificates \
    ccache \
    curl \
    git \
    unzip \
    zip \
    python3 \
    libglm-dev \
    libglfw3-dev \
    libgtest-dev \
    libvulkan-dev \
    glslang-tools \
    libfreetype-dev

add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get install -y --no-install-recommends gcc-13 g++-13

if ! cmake --version 2>/dev/null | head -1 | grep -q "${CMAKE_VERSION}"; then
    curl -fL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz" |
        tar -xz --strip-components=1 -C /usr/local
fi

echo "gcc: $(gcc-13 --version | head -1)"
echo "cmake: $(cmake --version | head -1)"
