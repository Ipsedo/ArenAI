cmake_minimum_required(VERSION 3.29)

project(arenai_core)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

# indicators
FetchContent_Declare(
        indicators
        GIT_REPOSITORY https://github.com/p-ranav/indicators.git
        GIT_TAG 3872f37abd90d7557bac5f834bfb45bd6c75259a # v2.3 + fixes
)
FetchContent_MakeAvailable(indicators)

##### ArenAI #####
file(GLOB_RECURSE ARENAI_CORE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_core SHARED ${ARENAI_CORE_SRC})
target_include_directories(arenai_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(
        arenai_core
        PUBLIC
        arenai_utils
        arenai_view
        arenai_model
        arenai_controller
        indicators
)
