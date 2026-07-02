cmake_minimum_required(VERSION 3.29)

project(arenai_utils)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE ARENAI_UTILS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_utils SHARED ${ARENAI_UTILS_SRC})
target_include_directories(arenai_utils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (ANDROID)
    target_link_libraries(arenai_utils android log)
endif ()
