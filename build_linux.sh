#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR="${SCRIPT_DIR}/build"

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir "${BUILD_DIR}"
fi

cd "${BUILD_DIR}"
cmake "${SCRIPT_DIR}"
make -j $(nproc)
