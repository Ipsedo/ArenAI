cmake_minimum_required(VERSION 3.11)

project(phyvr_rl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(PHYVR_RL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

include_directories(${PHYVR_RL_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_INCLUDE_DIRS})

set(PHYVR_RL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE PHYVR_RL_SRC "${PHYVR_RL_PATH}/*.cpp")

add_library(phyvr_rl SHARED ${PHYVR_RL_SRC})
