cmake_minimum_required(VERSION 3.29)

project(arenai_view_tests)

set(CMAKE_CXX_STANDARD 20)

enable_testing()
find_package(GTest REQUIRED)
# the tests exercise the GL renderer directly (glReadPixels, glClear, ...), so
# they link the GL entry points themselves — arenai_view keeps them PRIVATE.
# Same choice as arenai_view: mesa's libGLESv2 on Windows (the system opengl32
# only exports GL 1.1 and dispatches through WGL, not our mesa EGL contexts).
if (WIN32)
    if (NOT MESA_SDK_DIR)
        set(MESA_SDK_DIR "${CMAKE_SOURCE_DIR}/libs/mesa-devel")
    endif ()
    find_library(MESA_GLES_LIB libGLESv2 HINTS "${MESA_SDK_DIR}/lib/x64" REQUIRED)
    find_path(MESA_INCLUDE_DIR EGL/egl.h HINTS "${MESA_SDK_DIR}/include" REQUIRED)
    set(ARENAI_TEST_GL_LIB "${MESA_GLES_LIB}")
else ()
    find_package(OpenGL REQUIRED)
    set(ARENAI_TEST_GL_LIB OpenGL::GL)
endif ()

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(arenai_view_tests "${SRC}")

# arenai_view propagates the view/utils/controller headers through its PUBLIC interface.
target_link_libraries(arenai_view_tests GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json soil2 arenai_view "${ARENAI_TEST_GL_LIB}")
target_include_directories(arenai_view_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
# internal headers (shadow rendering is not part of the public API)
target_include_directories(arenai_view_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")

target_compile_definitions(arenai_view_tests PRIVATE
        ARENAI_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../resources")

if (ARENAI_REGENERATE_GOLDEN_IMAGES)
    target_compile_definitions(arenai_view_tests PRIVATE ARENAI_REGENERATE_GOLDEN_IMAGES)
endif ()

# fix opengl header with glfw on windows
if (WIN32)
    target_compile_definitions(arenai_view_tests PRIVATE GLFW_INCLUDE_NONE)
    # GL/gl.h + GL/glext.h come from the mesa SDK (the Windows SDK has no glext.h)
    target_include_directories(arenai_view_tests PRIVATE "${MESA_INCLUDE_DIR}")
endif ()

arenai_copy_runtime_dlls(arenai_view_tests)

add_test(
        NAME arenai_view_tests
        COMMAND arenai_view_tests
)
