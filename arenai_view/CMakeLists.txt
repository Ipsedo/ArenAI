cmake_minimum_required(VERSION 3.29)

project(arenai_view)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

##### GLM #####
find_package(glm REQUIRED)

# Embed the GLSL sources into the library: cmake/ArenaiEmbedShaders.cmake turns
# shaders/*.glsl into a generated header (EMBEDDED_SHADERS map) consumed by
# load_shader(). The built executables need no shader file at runtime.
file(GLOB ARENAI_VIEW_SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")
set(ARENAI_VIEW_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ARENAI_VIEW_EMBEDDED_SHADERS_H "${ARENAI_VIEW_GENERATED_DIR}/shaders_embedded.h")

add_custom_command(
        OUTPUT "${ARENAI_VIEW_EMBEDDED_SHADERS_H}"
        COMMAND ${CMAKE_COMMAND}
        -DSHADERS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/shaders
        -DOUTPUT_FILE=${ARENAI_VIEW_EMBEDDED_SHADERS_H}
        -P ${CMAKE_SOURCE_DIR}/cmake/ArenaiEmbedShaders.cmake
        DEPENDS ${ARENAI_VIEW_SHADER_FILES} "${CMAKE_SOURCE_DIR}/cmake/ArenaiEmbedShaders.cmake"
        COMMENT "Embedding GLSL shaders into shaders_embedded.h"
)

# ARENAI VIEW
file(GLOB_RECURSE ARENAI_VIEW_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_view SHARED ${ARENAI_VIEW_SRC} "${ARENAI_VIEW_EMBEDDED_SHADERS_H}")
target_include_directories(arenai_view PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(arenai_view PRIVATE "${ARENAI_VIEW_GENERATED_DIR}")

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

# Desktop OpenGL 3.3 core. On Linux libGL (Mesa/NVIDIA, dispatched by libglvnd)
# exports the full modern API as real symbols, so linking OpenGL::GL and
# declaring the extension prototypes (see src/opengl/gl.h) is enough — no
# function loader is needed. GL stays PRIVATE: it never leaks past arenai_view.
find_package(OpenGL REQUIRED)
target_link_libraries(arenai_view PRIVATE OpenGL::GL)

if (WIN32)
    # Mesa3D SDK (headers + import libs) from mesa-dist-win "devel" package,
    # fetched by arenai_desktop/install_dependencies.ps1 into libs/mesa-devel.
    # Only libEGL is needed here (context management); GL comes from OpenGL::GL.
    if (NOT MESA_SDK_DIR)
        set(MESA_SDK_DIR "${CMAKE_SOURCE_DIR}/libs/mesa-devel")
    endif ()
    find_path(MESA_INCLUDE_DIR EGL/egl.h HINTS "${MESA_SDK_DIR}/include" REQUIRED)
    find_library(MESA_EGL_LIB libEGL HINTS "${MESA_SDK_DIR}/lib/x64" REQUIRED)
    target_link_libraries(arenai_view PRIVATE "${MESA_EGL_LIB}")
    target_include_directories(arenai_view PRIVATE "${MESA_INCLUDE_DIR}")
else ()
    target_link_libraries(arenai_view PRIVATE EGL dl)
endif ()
