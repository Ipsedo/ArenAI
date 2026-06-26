cmake_minimum_required(VERSION 3.29)

project(arenai_view_tests)
enable_testing()
find_package(GTest REQUIRED)

set(ARENAI_VIEW_TESTS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
include_directories("${ARENAI_VIEW_TESTS_INCLUDE_DIRS}")

# add evo_motion_networks headers
file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(arenai_view_tests "${SRC}")

target_link_libraries(arenai_view_tests GTest::gtest GTest::gtest_main soil2 arenai_view)
target_include_directories(arenai_view_tests PRIVATE "${ARENAI_VIEW_INCLUDE_DIRS}" "${ARENAI_UTILS_INCLUDE_DIRS}" "${ARENAI_CONTROLLER_INCLUDE_DIRS}")

add_test(
        NAME arenai_view_tests
        COMMAND arenai_view_tests
)
