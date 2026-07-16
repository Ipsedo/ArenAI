cmake_minimum_required(VERSION 3.29)

project(arenai_controller)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE ARENAI_CONTROLLER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_controller SHARED ${ARENAI_CONTROLLER_SRC})
target_include_directories(arenai_controller PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
