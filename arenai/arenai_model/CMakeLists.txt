cmake_minimum_required(VERSION 3.29)

project(arenai_model)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

##### GLM #####
if (ANDROID)
    FetchContent_Declare(glm URL https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip)
    FetchContent_MakeAvailable(glm)
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
else ()
    find_package(Bullet REQUIRED)
endif ()

##### PhyVR #####
file(GLOB_RECURSE ARENAI_MODEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_model SHARED ${ARENAI_MODEL_SRC})
target_include_directories(arenai_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# glm::glm carries its own include directories.
target_link_libraries(
        arenai_model
        PUBLIC
        ${BULLET_LIBRARIES}
        arenai_utils
        arenai_view
        arenai_controller
        glm::glm
)

if (ANDROID)
    target_include_directories(arenai_model PUBLIC "${bullet3_SOURCE_DIR}/src")
else ()
    target_include_directories(arenai_model PUBLIC "${BULLET_INCLUDE_DIR}")
endif ()
