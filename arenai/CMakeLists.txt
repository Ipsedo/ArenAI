cmake_minimum_required(VERSION 3.29)

project(arenai)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ArenaiRuntimeDlls)

# Train main & lib
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_train")

# Desktop main (game)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_desktop")

# Tests
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_view/tests")

# fix opengl header with glfw on windows
if (WIN32)
    target_compile_definitions(arenai_desktop PRIVATE GLFW_INCLUDE_NONE)
    target_compile_definitions(arenai_view_tests PRIVATE GLFW_INCLUDE_NONE)
endif()
