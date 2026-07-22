cmake_minimum_required(VERSION 3.29)

project(arenai_train_tests)

set(CMAKE_CXX_STANDARD 20)

enable_testing()
find_package(GTest REQUIRED)

if(NOT DEFINED LIBTORCH_PATH)
    set(LIBTORCH_PATH "/opt/libtorch")
endif()
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(arenai_train_tests "${SRC}")

target_link_libraries(arenai_train_tests GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json soil2 ${TORCH_LIBRARIES} arenai_train_lib)
target_include_directories(arenai_train_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(arenai_train_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")

# fix opengl header with glfw on windows
if (WIN32)
    target_compile_definitions(arenai_train_tests PRIVATE GLFW_INCLUDE_NONE)
endif ()

arenai_copy_runtime_dlls(arenai_train_tests)

add_test(
        NAME arenai_train_tests
        COMMAND arenai_train_tests
)
