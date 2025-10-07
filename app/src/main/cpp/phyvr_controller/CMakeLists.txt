cmake_minimum_required(VERSION 3.29)

project(phyvr_controller)

include(FetchContent)

##### PhyVR #####
set(CMAKE_CXX_STANDARD 20)

set(PHYVR_CONTROLLER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
include_directories(${PHYVR_CONTROLLER_INCLUDE_DIRS})

file(GLOB_RECURSE PHYVR_CONTROLLER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(phyvr_controller SHARED ${PHYVR_CONTROLLER_SRC})
