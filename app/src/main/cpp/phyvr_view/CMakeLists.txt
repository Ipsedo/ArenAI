cmake_minimum_required(VERSION 3.5)

set(APP_NAME phyvr_view)

project(${APP_NAME})

include(FetchContent)

##### GLM #####
if (ANDROID)
    FetchContent_Declare(glm URL https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip)
    FetchContent_MakeAvailable(glm)
    include_directories("${glm_SOURCE_DIR}")
else ()
    find_package(glm REQUIRED)
endif ()

# PHYVR VIEW
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(PHYVR_VIEW_PATH_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

include_directories(${PHYVR_CONTROLLER_PATH_INCLUDE_DIRS})
include_directories(${PHYVR_UTILS_PATH_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_PATH_INCLUDE_DIRS})

set(PHYVR_VIEW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE PHYVR_VIEW_SRC "${PHYVR_VIEW_PATH}/*.cpp")
add_library(${APP_NAME} SHARED ${PHYVR_VIEW_SRC})

target_link_libraries(
        ${APP_NAME}
        EGL
        dl
        phyvr_utils
        glm::glm
)

if (ANDROID)
    target_link_libraries(${APP_NAME} GLESv3)
else ()
    target_link_libraries(${APP_NAME} GLESv2)
endif ()
