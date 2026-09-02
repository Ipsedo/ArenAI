cmake_minimum_required(VERSION 3.29)

project(arenai_desktop_tests)

set(CMAKE_CXX_STANDARD 20)

enable_testing()
find_package(GTest REQUIRED)

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# arenai_desktop is an executable, not a library: the sources under test are
# compiled straight into the test binary
set(ARENAI_DESKTOP_TESTED_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/game_environment.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/user_preferences.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/bindings.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/gamepad.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/controller/mouse_keyboard.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/gui/rml/input.cpp")

add_executable(arenai_desktop_tests "${SRC}" ${ARENAI_DESKTOP_TESTED_SRC})

target_link_libraries(
        arenai_desktop_tests
        PRIVATE
        GTest::gtest
        GTest::gtest_main
        nlohmann_json::nlohmann_json
        arenai_model
        arenai_view
        arenai_utils
        arenai_core
        arenai_agent
        RmlUi::Core
)
target_include_directories(arenai_desktop_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(arenai_desktop_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")

target_compile_definitions(arenai_desktop_tests PRIVATE
        ARENAI_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../resources")

if (ARENAI_REGENERATE_GOLDEN_IMAGES)
    target_compile_definitions(arenai_desktop_tests PRIVATE ARENAI_REGENERATE_GOLDEN_IMAGES)
endif ()

# fix opengl header with glfw on windows
if (WIN32)
    target_compile_definitions(arenai_desktop_tests PRIVATE GLFW_INCLUDE_NONE)
endif ()

arenai_copy_runtime_dlls(arenai_desktop_tests)

add_test(
        NAME arenai_desktop_tests
        COMMAND arenai_desktop_tests
)
