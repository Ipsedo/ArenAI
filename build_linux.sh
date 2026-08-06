#!/usr/bin/env bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR="${SCRIPT_DIR}/build"

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir "${BUILD_DIR}"
fi

cd "${BUILD_DIR}"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3" "${SCRIPT_DIR}"
make -j $(nproc)
