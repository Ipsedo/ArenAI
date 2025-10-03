cmake_minimum_required(VERSION 3.11)

project(phyvr_core)

include(FetchContent)

##### PhyVR #####
set(CMAKE_CXX_STANDARD 20)

include_directories(${PHYVR_UTILS_INCLUDE_DIRS})
include_directories(${PHYVR_CONTROLLER_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_INCLUDE_DIRS})
include_directories(${PHYVR_MODEL_INCLUDE_DIRS})

set(PHYVR_CORE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
include_directories(${PHYVR_CORE_INCLUDE_DIRS})

file(GLOB_RECURSE PHYVR_CORE_SRC "${PHYVR_CPP_FOLDER}/phyvr_core/src/*.cpp")

add_library(phyvr_core SHARED ${PHYVR_CORE_SRC})

target_link_libraries(
        phyvr_core
        phyvr_utils
        phyvr_view
        phyvr_model
)
