cmake_minimum_required(VERSION 3.29)

project(arenai_utils)

set(CMAKE_CXX_STANDARD 20)

set(ARENAI_UTILS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
include_directories(${ARENAI_UTILS_INCLUDE_DIRS})

file(GLOB_RECURSE ARENAI_UTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_utils SHARED ${ARENAI_UTILS_SRC})

if (ANDROID)
    target_link_libraries(arenai_utils android log)
endif ()
