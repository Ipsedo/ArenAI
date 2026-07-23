cmake_minimum_required(VERSION 3.29)

project(arenai)

set(CMAKE_CXX_STANDARD 20)

enable_testing()

# Auto-export all symbols from our shared libraries on Windows (replaces the
# per-target WINDOWS_EXPORT_ALL_SYMBOLS property in each module).
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Windows has no rpath: executables only find DLLs next to themselves. One
# shared runtime directory for every target — ours and FetchContent's (RmlUi)
# alike — means no per-target DLL copying, and vcpkg's applocal deployment
# (freetype and friends next to rmlui.dll) lands where the executables live.
if (WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ArenaiRuntimeDlls)

# Force golden-image tests to overwrite their reference files instead of
# comparing against them. ON disables every pixel assertion, so never turn it on
# to get a red CI green: that only hides the diff.
# The committed goldens are llvmpipe/Ubuntu-specific — regenerate them with
# scripts/goldens_docker.sh --regenerate, which runs the CI render environment.
option(ARENAI_REGENERATE_GOLDEN_IMAGES
        "Rebuild golden reference images instead of comparing against them" OFF)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_controller")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_utils")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_view")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_model")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_core")

# Train main & lib
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_agent")

# Desktop main (game)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_desktop")

# Tests
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_view/tests")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_agent/tests")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_model/tests")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_core/tests")
