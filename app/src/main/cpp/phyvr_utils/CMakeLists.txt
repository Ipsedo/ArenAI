cmake_minimum_required(VERSION 3.11)

set(APP_NAME phyvr_utils)

project(${APP_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(PHYVR_UTILS_PATH_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
include_directories(${PHYVR_UTILS_PATH_INCLUDE_DIRS})

set(PHYVR_UTILS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE PHYVR_UTILS_SRC "${PHYVR_UTILS_PATH}/*.cpp")
add_library(${APP_NAME} SHARED ${PHYVR_UTILS_SRC})

target_link_libraries(
        ${APP_NAME}
        android
        log
        dl
        jnigraphics
)
