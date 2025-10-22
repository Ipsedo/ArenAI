cmake_minimum_required(VERSION 3.29)

project(arenai_view)

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

# ARENAI VIEW
set(CMAKE_CXX_STANDARD 20)

set(ARENAI_VIEW_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

include_directories(${ARENAI_CONTROLLER_INCLUDE_DIRS})
include_directories(${ARENAI_UTILS_INCLUDE_DIRS})
include_directories(${ARENAI_VIEW_INCLUDE_DIRS})

file(GLOB_RECURSE ARENAI_VIEW_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_view SHARED ${ARENAI_VIEW_SRC})

target_link_libraries(
        arenai_view
        EGL
        dl
        arenai_utils
        glm::glm
)

if (ANDROID)
    target_link_libraries(arenai_view GLESv3)
    target_include_directories(arenai_view PUBLIC "${glm_SOURCE_DIR}")
else ()
    target_link_libraries(arenai_view GLESv2)
    target_include_directories(arenai_view PUBLIC "${GLM_INCLUDE_DIRS}")
endif ()
