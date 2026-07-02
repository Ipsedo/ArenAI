# arenai_copy_runtime_dlls(<target> [TORCH])
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

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> "${out_dir}"
        COMMAND_EXPAND_LISTS
    )

    if(ARG_TORCH)
        if(EXISTS "${LIBTORCH_PATH}/lib")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory "${LIBTORCH_PATH}/lib" "${out_dir}"
            )
        else()
            message(WARNING "LIBTORCH_PATH/lib not found (${LIBTORCH_PATH}/lib); run install_dependencies.ps1")
        endif()
    endif()

    if(EXISTS "${MESA_PATH}")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${MESA_PATH}" "${out_dir}"
        )
    else()
        message(WARNING "MESA_PATH not found (${MESA_PATH}); run install_dependencies.ps1 to fetch Mesa3D DLLs")
    endif()
endfunction()
