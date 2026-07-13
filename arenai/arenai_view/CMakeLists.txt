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
