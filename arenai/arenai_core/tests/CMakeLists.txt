cmake_minimum_required(VERSION 3.29)

project(arenai_core_tests)

set(CMAKE_CXX_STANDARD 20)

enable_testing()
find_package(GTest REQUIRED)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(arenai_core_tests "${SRC}")

target_link_libraries(arenai_core_tests GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json soil2 arenai_core)
target_include_directories(arenai_core_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(arenai_core_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")

target_compile_definitions(arenai_core_tests PRIVATE
        ARENAI_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../android/app/src/main/assets")

# fix opengl header with glfw on windows
if (WIN32)
    target_compile_definitions(arenai_core_tests PRIVATE GLFW_INCLUDE_NONE)
endif ()

arenai_copy_runtime_dlls(arenai_core_tests)

add_test(
        NAME arenai_core_tests
        COMMAND arenai_core_tests
)
