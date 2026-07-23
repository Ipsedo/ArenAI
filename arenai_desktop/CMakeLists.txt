cmake_minimum_required(VERSION 3.29)

# Must be set before project() so CMake picks it up at configure time
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/libs/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(arenai_desktop)

set(CMAKE_CXX_STANDARD 20)

# argparse
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v3.2
)
FetchContent_MakeAvailable(argparse)

# ArenAI Desktop
set(ARENAI_DESKTOP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE ARENAI_DESKTOP_SRC "${ARENAI_DESKTOP_PATH}/*.cpp")

add_executable(arenai_desktop ${ARENAI_DESKTOP_SRC})

# ArenAI module headers propagate through arenai_agent's PUBLIC interface.
# Windowing (GLFW) is fully encapsulated inside arenai_view.
# RmlUi is reserved to the src/gui/ hexagon: only gui/*.cpp may include it,
# every other folder goes through gui/menu.h (which exposes no RmlUi type).
target_link_libraries(
        arenai_desktop
        PRIVATE
        arenai_model
        arenai_view
        arenai_utils
        arenai_core
        arenai_agent
        RmlUi::Core
        # user preferences are (de)serialized to JSON in src/core/
        nlohmann_json::nlohmann_json
        argparse
)

# this is not a leak, just an arg ON/OFF
arenai_copy_runtime_dlls(arenai_desktop TORCH)

# Game assets copied next to the executable: main.cpp resolves the resources
# folder from the binary's location, whatever the working directory.
# A standalone ALL target rather than a POST_BUILD command: POST_BUILD only
# fires when the executable relinks, so asset-only edits (RML/RCSS, textures)
# would never reach the build tree; this target runs on every build and
# copy_directory_if_different keeps the no-change case cheap.
add_custom_target(
        arenai_desktop_assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${CMAKE_SOURCE_DIR}/resources" "$<TARGET_FILE_DIR:arenai_desktop>/resources"
)
# the executable depends on the assets (not the other way round), so building
# just the arenai_desktop target — what an IDE Run configuration does — also
# refreshes the copied resources
add_dependencies(arenai_desktop arenai_desktop_assets)
