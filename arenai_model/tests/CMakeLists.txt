cmake_minimum_required(VERSION 3.29)

project(arenai_model_tests)

set(CMAKE_CXX_STANDARD 20)

enable_testing()
find_package(GTest REQUIRED)

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(arenai_model_tests "${SRC}")

target_link_libraries(arenai_model_tests GTest::gtest GTest::gtest_main arenai_model)
target_include_directories(arenai_model_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(arenai_model_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")

target_compile_definitions(arenai_model_tests PRIVATE
        ARENAI_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../resources")

# fix opengl header with glfw on windows
if (WIN32)
    target_compile_definitions(arenai_model_tests PRIVATE GLFW_INCLUDE_NONE)
endif ()

arenai_copy_runtime_dlls(arenai_model_tests)

add_test(
        NAME arenai_model_tests
        COMMAND arenai_model_tests
)
