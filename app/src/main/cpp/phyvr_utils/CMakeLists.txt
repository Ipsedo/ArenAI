cmake_minimum_required(VERSION 3.11)

project(phyvr_utils)

set(CMAKE_CXX_STANDARD 20)

set(PHYVR_UTILS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
include_directories(${PHYVR_UTILS_INCLUDE_DIRS})

file(GLOB_RECURSE PHYVR_UTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(phyvr_utils SHARED ${PHYVR_UTILS_SRC})

if (ANDROID)
    target_link_libraries(phyvr_utils android log)
endif ()
