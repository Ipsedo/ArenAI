# arenai_copy_runtime_dlls(<target> [TORCH])
#
# Copy, next to <target>'s executable on Windows, the runtime DLLs it needs:
#   - everything in the target's link graph (arenai_*, glfw, gtest, torch core)
#     is resolved automatically via $<TARGET_RUNTIME_DLLS>;
#   - LibTorch's loose DLLs (OpenMP, libuv, ...) are not CMake targets, so they
#     are copied from libs/libtorch/lib, but only when the TORCH keyword is given;
#   - Mesa3D is loaded at runtime by the OpenGL/EGL ICD (not in the link graph)
#     and must sit next to the exe, so it is copied wholesale.
#
# POST_BUILD => the copy runs whenever the exe is (re)built, and the call must
# live in the same CMakeLists.txt that created <target>. No-op on non-Windows.

function(arenai_copy_runtime_dlls target)
    cmake_parse_arguments(ARG "TORCH" "" "" ${ARGN})

    if(NOT WIN32)
        return()
    endif()
    if(NOT LIBTORCH_PATH)
        set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch")
    endif()
    if(NOT MESA_PATH)
        set(MESA_PATH "${CMAKE_SOURCE_DIR}/libs/mesa/x64")
    endif()

    set(out_dir "$<TARGET_FILE_DIR:${target}>")

    # Exactly the DLLs this target links, resolved from the link graph.
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> "${out_dir}"
        COMMAND_EXPAND_LISTS
    )

    # LibTorch's loose runtime deps (libiomp5md, uv, ...) aren't CMake targets,
    # so $<TARGET_RUNTIME_DLLS> can't see them: copy them only where Torch is used.
    if(ARG_TORCH)
        if(EXISTS "${LIBTORCH_PATH}/lib")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory "${LIBTORCH_PATH}/lib" "${out_dir}"
            )
        else()
            message(WARNING "LIBTORCH_PATH/lib not found (${LIBTORCH_PATH}/lib); run install_dependencies.ps1")
        endif()
    endif()

    # Mesa3D runtime: loaded by the GL/EGL ICD, not auto-resolvable, must be adjacent.
    if(EXISTS "${MESA_PATH}")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${MESA_PATH}" "${out_dir}"
        )
    else()
        message(WARNING "MESA_PATH not found (${MESA_PATH}); run install_dependencies.ps1 to fetch Mesa3D DLLs")
    endif()
endfunction()
