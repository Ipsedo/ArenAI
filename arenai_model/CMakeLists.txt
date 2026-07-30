cmake_minimum_required(VERSION 3.29)

project(arenai_model)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

##### GLM #####
if (ANDROID)
    FetchContent_Declare(glm URL https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip)
    FetchContent_MakeAvailable(glm)
else ()
    find_package(glm REQUIRED)
endif ()

##### Jolt Physics #####
set(TARGET_UNIT_TESTS OFF CACHE INTERNAL "turn off jolt tests")
set(TARGET_HELLO_WORLD OFF CACHE INTERNAL "turn off jolt hello world")
set(TARGET_PERFORMANCE_TEST OFF CACHE INTERNAL "turn off jolt performance test")
set(TARGET_SAMPLES OFF CACHE INTERNAL "turn off jolt samples")
set(TARGET_VIEWER OFF CACHE INTERNAL "turn off jolt viewer")
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE INTERNAL "turn off jolt lto")

FetchContent_Declare(
        JoltPhysics
        GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
        GIT_TAG v5.3.0
        SOURCE_SUBDIR Build
)
FetchContent_MakeAvailable(JoltPhysics)

# Jolt is a static library linked into this shared library
set_target_properties(Jolt PROPERTIES POSITION_INDEPENDENT_CODE ON)

##### ArenAI #####
file(GLOB_RECURSE ARENAI_MODEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_model SHARED ${ARENAI_MODEL_SRC})
target_include_directories(arenai_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(arenai_model PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(
        arenai_model
        PUBLIC
        arenai_utils
        arenai_view
        arenai_controller
        glm::glm
        PRIVATE
        Jolt
)
