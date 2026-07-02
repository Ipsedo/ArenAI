cmake_minimum_required(VERSION 3.29)

project(arenai_view_tests)

set(CMAKE_CXX_STANDARD 20)

enable_testing()
find_package(GTest REQUIRED)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(arenai_view_tests "${SRC}")

# arenai_view propagates the view/utils/controller headers through its PUBLIC interface.
target_link_libraries(arenai_view_tests GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json soil2 arenai_view)
target_include_directories(arenai_view_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_definitions(arenai_view_tests PRIVATE
        ARENAI_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../android/app/src/main/assets")

# fix opengl header with glfw on windows
if (WIN32)
    target_compile_definitions(arenai_view_tests PRIVATE GLFW_INCLUDE_NONE)
endif ()

arenai_copy_runtime_dlls(arenai_view_tests)

add_test(
        NAME arenai_view_tests
        COMMAND arenai_view_tests
)
