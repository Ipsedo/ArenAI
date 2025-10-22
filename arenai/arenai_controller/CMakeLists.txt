cmake_minimum_required(VERSION 3.29)

project(arenai_controller)

include(FetchContent)

##### PhyVR #####
set(CMAKE_CXX_STANDARD 20)

set(ARENAI_CONTROLLER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
include_directories(${ARENAI_CONTROLLER_INCLUDE_DIRS})

file(GLOB_RECURSE ARENAI_CONTROLLER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(arenai_controller SHARED ${ARENAI_CONTROLLER_SRC})
