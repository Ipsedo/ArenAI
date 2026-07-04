#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if ! [ -d "${SCRIPT_DIR}/build"]; then
  mkdir build
fi

cd "${SCRIPT_DIR}/build"
cmake ${SCRIPT_DIR}
make -j $(nproc)
