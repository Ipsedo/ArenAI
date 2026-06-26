cmake_minimum_required(VERSION 3.11)

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

if(WIN32)
    add_custom_command(TARGET arenai_desktop POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:arenai_desktop>
            $<TARGET_FILE_DIR:arenai_desktop>
        COMMAND_EXPAND_LISTS
    )
    # Copy all LibTorch DLLs (includes transitive deps like fbgemm, uv, libiomp5md)
    add_custom_command(TARGET arenai_desktop POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${LIBTORCH_PATH}/lib"
            $<TARGET_FILE_DIR:arenai_desktop>
    )
    # Copy vcpkg DLLs (zlib, bullet, glfw, glm, etc.)
    if(NOT VCPKG_INSTALLED_DIR)
        set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/libs/vcpkg/installed")
    endif()
    add_custom_command(TARGET arenai_desktop POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${VCPKG_INSTALLED_DIR}/x64-windows/bin"
            $<TARGET_FILE_DIR:arenai_desktop>
    )
    # Copy prebuilt Mesa3D runtime DLLs (GLES/EGL/gallium: llvmpipe, d3d12, zink)
    # downloaded by install_dependencies.ps1 into libs/mesa/x64.
    if(NOT MESA_PATH)
        set(MESA_PATH "${CMAKE_SOURCE_DIR}/libs/mesa/x64")
    endif()
    if(EXISTS "${MESA_PATH}")
        add_custom_command(TARGET arenai_desktop POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${MESA_PATH}"
                $<TARGET_FILE_DIR:arenai_desktop>
        )
    else()
        message(WARNING "MESA_PATH not found (${MESA_PATH}); run install_dependencies.ps1 to fetch Mesa3D DLLs")
    endif()
endif()
