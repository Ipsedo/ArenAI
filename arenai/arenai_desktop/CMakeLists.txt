cmake_minimum_required(VERSION 3.29)

# Must be set before project() so CMake picks it up at configure time
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/libs/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(arenai_desktop)

set(CMAKE_CXX_STANDARD 20)

# ArenAI modules
include_directories(${ARENAI_CONTROLLER_INCLUDE_DIRS})
include_directories(${ARENAI_UTILS_INCLUDE_DIRS})
include_directories(${ARENAI_VIEW_INCLUDE_DIRS})
include_directories(${ARENAI_MODEL_INCLUDE_DIRS})
include_directories(${ARENAI_CORE_INCLUDE_DIRS})
include_directories(${ARENAI_TRAIN_INCLUDE_DIRS})

# dependencies
find_package(glfw3 REQUIRED)

# ArenAI Desktop
set(ARENAI_DESKTOP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE ARENAI_DESKTOP_SRC "${ARENAI_DESKTOP_PATH}/*.cpp")

add_executable(arenai_desktop ${ARENAI_DESKTOP_SRC})

target_link_libraries(
        arenai_desktop
        arenai_train_lib
        glfw
)

arenai_copy_runtime_dlls(arenai_desktop TORCH)
