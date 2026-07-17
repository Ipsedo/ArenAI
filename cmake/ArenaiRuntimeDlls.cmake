# arenai_copy_runtime_dlls(<target> [TORCH])
#
# Windows-only DLL deployment. Every executable and DLL of the build — ours and
# FetchContent's (RmlUi) alike — lands in the shared CMAKE_RUNTIME_OUTPUT_DIRECTORY
# (see the top-level CMakeLists), and vcpkg's applocal step deploys the vcpkg
# DLLs (glfw, gtest, RmlUi's freetype, ...) next to each binary it links.
# What remains are the non-CMake SDKs: libtorch, and optionally the Mesa3D
# runtime (its lavapipe/dozen Vulkan ICDs, a software fallback for GPU-less
# machines — see install_dependencies.ps1 -SoftwareVulkan). Those are copied
# by shared custom targets created once — per-target POST_BUILD copies into
# the same directory would race each other and the applocal scans under a
# parallel ninja.
function(arenai_copy_runtime_dlls target)
    cmake_parse_arguments(ARG "TORCH" "" "" ${ARGN})

    if(NOT WIN32)
        return()
    endif()

    if(NOT TARGET arenai_mesa_dlls)
        if(NOT MESA_PATH)
            set(MESA_PATH "${CMAKE_SOURCE_DIR}/libs/mesa/x64")
        endif()
        # optional: the primary Vulkan path is the system vulkan-1.dll and the
        # GPU driver's ICD, no warning when the Mesa bundle is absent
        if(EXISTS "${MESA_PATH}")
            add_custom_target(arenai_mesa_dlls
                COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                    "${MESA_PATH}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            )
        endif()
    endif()
    if(TARGET arenai_mesa_dlls)
        add_dependencies(${target} arenai_mesa_dlls)
    endif()

    if(ARG_TORCH)
        if(NOT TARGET arenai_torch_dlls)
            if(NOT LIBTORCH_PATH)
                set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch")
            endif()
            if(EXISTS "${LIBTORCH_PATH}/lib")
                add_custom_target(arenai_torch_dlls
                    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                        "${LIBTORCH_PATH}/lib" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                )
            else()
                message(WARNING "LIBTORCH_PATH/lib not found (${LIBTORCH_PATH}/lib); run install_dependencies.ps1")
            endif()
        endif()
        if(TARGET arenai_torch_dlls)
            add_dependencies(${target} arenai_torch_dlls)
        endif()
    endif()
endfunction()
