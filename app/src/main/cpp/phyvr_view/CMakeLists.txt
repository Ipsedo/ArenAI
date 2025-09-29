cmake_minimum_required(VERSION 3.5)

project(phyvr_view)

include(FetchContent)

##### GLM #####
if (ANDROID)
    FetchContent_Declare(glm URL https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip)
    FetchContent_MakeAvailable(glm)
    include_directories("${glm_SOURCE_DIR}")
else ()
    find_package(glm REQUIRED)
    include_directories(${GLM_INCLUDE_DIRS})
endif ()

# PHYVR VIEW
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(PHYVR_VIEW_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

include_directories(${PHYVR_CONTROLLER_INCLUDE_DIRS})
include_directories(${PHYVR_UTILS_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_INCLUDE_DIRS})

file(GLOB_RECURSE PHYVR_VIEW_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(phyvr_view SHARED ${PHYVR_VIEW_SRC})

target_link_libraries(
        phyvr_view
        EGL
        dl
        phyvr_utils
        glm::glm
)

if (ANDROID)
    target_link_libraries(phyvr_view GLESv3)
    target_include_directories(phyvr_view PUBLIC "${glm_SOURCE_DIR}")
else ()
    target_link_libraries(phyvr_view GLESv2)
    target_include_directories(phyvr_view PUBLIC "${GLM_INCLUDE_DIRS}")
endif ()
