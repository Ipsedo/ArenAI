cmake_minimum_required(VERSION 3.11)

set(APP_NAME phyvr)

project(${APP_NAME})

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

include_directories(${PHYVR_UTILS_PATH_INCLUDE_DIRS})
include_directories(${PHYVR_CONTROLLER_PATH_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_PATH_INCLUDE_DIRS})
include_directories(${PHYVR_MODEL_PATH_INCLUDE_DIRS})
include_directories(${PHYVR_RL_PATH_INCLUDE_DIRS})

if (ANDROID)
    include(AndroidNdkModules)
    android_ndk_import_module_native_app_glue()

    file(GLOB_RECURSE PHYVR_SRC "${PHYVR_PATH}/*.cpp")

    add_library(${APP_NAME} SHARED ${PHYVR_SRC})

    target_link_libraries(
            ${APP_NAME}
            -Wl,--whole-archive native_app_glue -Wl,--no-whole-archive
            android
            # phyvr
            phyvr_utils
            phyvr_view
            phyvr_model
    )

else ()
    file(GLOB_RECURSE PHYVR_SRC "${PHYVR_PATH}/train/*.cpp")

    add_executable(${APP_NAME} ${PHYVR_SRC})

    target_link_libraries(
            ${APP_NAME}
            phyvr_utils
            phyvr_view
            phyvr_model
            phyvr_rl
    )
endif ()
