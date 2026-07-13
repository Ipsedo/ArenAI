cmake_minimum_required(VERSION 3.29)

project(arenai_view)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

##### GLM #####
find_package(glm REQUIRED)

# arenai_copy_shaders(<target>)
# Copies the GLSL sources next to <target>'s executable, where load_shader()
# resolves them at runtime ("<exe dir>/shaders"). Call it on every executable
# that renders (game, training, tests).
set(ARENAI_VIEW_SHADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders" CACHE INTERNAL "")
function(arenai_copy_shaders target)
    add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ARENAI_VIEW_SHADERS_DIR}" "$<TARGET_FILE_DIR:${target}>/shaders"
    )
endfunction()

# ARENAI VIEW
file(GLOB_RECURSE ARENAI_VIEW_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_view SHARED ${ARENAI_VIEW_SRC})
target_include_directories(arenai_view PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# glm::glm carries its own include directories, so no manual include path is needed.
target_link_libraries(
        arenai_view
        PUBLIC
        arenai_utils
        arenai_controller
        glm::glm
)

find_package(glfw3 REQUIRED)
target_link_libraries(arenai_view PRIVATE glfw)

if (WIN32)
    # Mesa3D SDK (headers + import libs) from mesa-dist-win "devel" package,
    # fetched by arenai_desktop/install_dependencies.ps1 into libs/mesa-devel.
    if (NOT MESA_SDK_DIR)
        set(MESA_SDK_DIR "${CMAKE_SOURCE_DIR}/libs/mesa-devel")
    endif ()
    find_path(MESA_INCLUDE_DIR GLES3/gl3.h HINTS "${MESA_SDK_DIR}/include" REQUIRED)
    find_library(MESA_EGL_LIB libEGL HINTS "${MESA_SDK_DIR}/lib/x64" REQUIRED)
    find_library(MESA_GLES2_LIB libGLESv2 HINTS "${MESA_SDK_DIR}/lib/x64" REQUIRED)
    target_link_libraries(arenai_view PUBLIC "${MESA_EGL_LIB}" "${MESA_GLES2_LIB}")
    target_include_directories(arenai_view PUBLIC "${MESA_INCLUDE_DIR}")
else ()
    target_link_libraries(arenai_view PUBLIC EGL GLESv2 dl)
endif ()
