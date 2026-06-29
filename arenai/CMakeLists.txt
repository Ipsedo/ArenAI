cmake_minimum_required(VERSION 3.29)

project(arenai)

set(CMAKE_CXX_STANDARD 20)

# Auto-export all symbols from our shared libraries on Windows (replaces the
# per-target WINDOWS_EXPORT_ALL_SYMBOLS property in each module).
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ArenaiRuntimeDlls)

# Train main & lib
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_train")

# Desktop main (game)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_desktop")

# Tests
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_view/tests")
