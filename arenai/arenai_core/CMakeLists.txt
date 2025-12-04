cmake_minimum_required(VERSION 3.29)

project(arenai_core)

include(FetchContent)

# indicators
FetchContent_Declare(
        indicators
        GIT_REPOSITORY https://github.com/p-ranav/indicators.git
)
FetchContent_MakeAvailable(indicators)

##### PhyVR #####
set(CMAKE_CXX_STANDARD 20)

include_directories(${ARENAI_UTILS_INCLUDE_DIRS})
include_directories(${ARENAI_CONTROLLER_INCLUDE_DIRS})
include_directories(${ARENAI_VIEW_INCLUDE_DIRS})
include_directories(${ARENAI_MODEL_INCLUDE_DIRS})

set(ARENAI_CORE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
include_directories(${ARENAI_CORE_INCLUDE_DIRS})

file(GLOB_RECURSE ARENAI_CORE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(arenai_core SHARED ${ARENAI_CORE_SRC})

target_link_libraries(
        arenai_core
        arenai_utils
        arenai_view
        arenai_model
        indicators
)
