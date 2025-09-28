cmake_minimum_required(VERSION 3.11)

project(phyvr)

include(FetchContent)

##### PhyVR #####
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_utils")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_controller")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_view")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_model")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_rl")

set(PHYVR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

include_directories(${PHYVR_UTILS_INCLUDE_DIRS})
include_directories(${PHYVR_CONTROLLER_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_INCLUDE_DIRS})
include_directories(${PHYVR_MODEL_INCLUDE_DIRS})
include_directories(${PHYVR_RL_INCLUDE_DIRS})

if (ANDROID)
    include(AndroidNdkModules)
    android_ndk_import_module_native_app_glue()

    file(GLOB_RECURSE PHYVR_SRC "${PHYVR_PATH}/*.cpp")

    add_library(phyvr SHARED ${PHYVR_SRC})

    target_link_libraries(
            phyvr
            -Wl,--whole-archive native_app_glue -Wl,--no-whole-archive
            android
            jnigraphics
    )

else ()
    file(GLOB_RECURSE PHYVR_SRC "${PHYVR_PATH}/train/*.cpp")

    add_executable(phyvr ${PHYVR_SRC})
endif ()

target_link_libraries(
        phyvr
        phyvr_utils
        phyvr_view
        phyvr_model
        phyvr_rl
)
