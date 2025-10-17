cmake_minimum_required(VERSION 3.29)

project(arenai_model)

include(FetchContent)

##### GLM #####
if (ANDROID)
    FetchContent_Declare(glm URL https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip)
    FetchContent_MakeAvailable(glm)
    include_directories("${glm_SOURCE_DIR}")
else ()
    find_package(glm REQUIRED)
endif ()

##### Bullet3 #####
if (ANDROID)
    set(BUILD_UNIT_TESTS OFF CACHE INTERNAL "turn off bullet test")
    set(BUILD_CPU_DEMOS OFF CACHE INTERNAL "turn off bullet demos")
    set(BUILD_PYBULLET OFF CACHE INTERNAL "turn off pybullet")
    set(BUILD_ENET OFF CACHE INTERNAL "turn off bullet enet")
    set(BUILD_CLSOCKET OFF CACHE INTERNAL "turn off bullet enet")
    set(BUILD_EGL OFF CACHE INTERNAL "turn off bullet EGL")
    set(BUILD_OPENGL3_DEMOS OFF CACHE INTERNAL "turn off bullet opengl demos")
    set(BUILD_BULLET2_DEMOS OFF CACHE INTERNAL "turn off bullet2 demos")
    set(BUILD_EXTRAS OFF CACHE INTERNAL "turn off bullet extra")
    set(BULLET2_MULTITHREADING ON CACHE INTERNAL "turn off bullet extra")

    FetchContent_Declare(
            bullet3
            GIT_REPOSITORY https://github.com/bulletphysics/bullet3.git
            GIT_TAG        3.25 # release 3.25
    )
    FetchContent_MakeAvailable(bullet3)
    include_directories("${bullet3_SOURCE_DIR}/src")
else ()
    find_package(Bullet REQUIRED)
    include_directories("${BULLET_INCLUDE_DIR}")
endif ()

##### PhyVR #####
set(CMAKE_CXX_STANDARD 20)

set(ARENAI_MODEL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

include_directories(${ARENAI_CONTROLLER_INCLUDE_DIRS})
include_directories(${ARENAI_UTILS_INCLUDE_DIRS})
include_directories(${ARENAI_VIEW_INCLUDE_DIRS})

include_directories(${ARENAI_MODEL_INCLUDE_DIRS})

file(GLOB_RECURSE ARENAI_MODEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_model SHARED ${ARENAI_MODEL_SRC})

target_link_libraries(
        arenai_model
        BulletDynamics BulletCollision LinearMath
        arenai_utils
        arenai_view
        glm::glm
)

if (ANDROID)
    target_include_directories(arenai_model PUBLIC "${bullet3_SOURCE_DIR}/src")
    target_include_directories(arenai_model PUBLIC "${glm_SOURCE_DIR}")
else ()
    target_include_directories(arenai_model PUBLIC "${BULLET_INCLUDE_DIR}")
    target_include_directories(arenai_model PUBLIC "${GLM_INCLUDE_DIR}")
endif ()

target_include_directories(arenai_model PUBLIC "${ARENAI_UTILS_INCLUDE_DIRS}")
target_include_directories(arenai_model PUBLIC "${ARENAI_VIEW_INCLUDE_DIRS}")
