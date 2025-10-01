cmake_minimum_required(VERSION 3.11)

project(phyvr)

include(FetchContent)

##### PhyVR #####
set(CMAKE_CXX_STANDARD 20)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_utils")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_controller")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_view")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_model")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/phyvr_rl")

include_directories(${PHYVR_UTILS_INCLUDE_DIRS})
include_directories(${PHYVR_CONTROLLER_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_INCLUDE_DIRS})
include_directories(${PHYVR_MODEL_INCLUDE_DIRS})
include_directories(${PHYVR_RL_INCLUDE_DIRS})

if (ANDROID)
    include(AndroidNdkModules)
    android_ndk_import_module_native_app_glue()

    file(GLOB_RECURSE PHYVR_ANDROID_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/android/*.cpp")
    file(GLOB_RECURSE PHYVR_CORE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
    set(PHYVR_SRC ${PHYVR_ANDROID_SRC} ${PHYVR_CORE_SRC})

    add_library(phyvr SHARED ${PHYVR_SRC})

    target_link_libraries(
            phyvr
            -Wl,--whole-archive native_app_glue -Wl,--no-whole-archive
            android
            jnigraphics
    )

else ()
    file(GLOB_RECURSE PHYVR_TRAIN_SRC "${PHYVR_PATH}/train/*.cpp")
    file(GLOB_RECURSE PHYVR_CORE_SRC "${PHYVR_PATH}/core/*.cpp")
    set(PHYVR_SRC ${PHYVR_TRAIN_SRC} ${PHYVR_CORE_SRC})

    add_executable(phyvr ${PHYVR_SRC})
endif ()

target_link_libraries(
        phyvr
        phyvr_utils
        phyvr_view
        phyvr_model
        phyvr_rl
)
