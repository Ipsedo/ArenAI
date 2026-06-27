cmake_minimum_required(VERSION 3.11)

project(arenai)

set(CMAKE_CXX_STANDARD 20)

# Runtime-DLL copy helper: defines arenai_copy_runtime_dlls(<target>), called
# from each subdirectory after its executable is defined (POST_BUILD must live
# in the target's own directory). Included here so all subdirs can use it.
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
