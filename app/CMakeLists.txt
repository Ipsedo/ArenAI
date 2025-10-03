cmake_minimum_required(VERSION 3.11)

project(phyvr)

include(FetchContent)

include(AndroidNdkModules)
android_ndk_import_module_native_app_glue()

##### PhyVR #####
set(CMAKE_CXX_STANDARD 20)

set(PHYVR_CPP_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp")

add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_utils")
add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_controller")
add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_view")
add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_model")
add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_core")

include_directories(${PHYVR_UTILS_INCLUDE_DIRS})
include_directories(${PHYVR_CONTROLLER_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_INCLUDE_DIRS})
include_directories(${PHYVR_MODEL_INCLUDE_DIRS})
include_directories(${PHYVR_CORE_INCLUDE_DIRS})

file(GLOB_RECURSE PHYVR_ANDROID_SRC "${PHYVR_CPP_FOLDER}/src/*.cpp")

add_library(phyvr SHARED ${PHYVR_ANDROID_SRC})

target_link_libraries(
        phyvr
        -Wl,--whole-archive native_app_glue -Wl,--no-whole-archive
        android
        jnigraphics
        phyvr_utils
        phyvr_view
        phyvr_model
        phyvr_core
)
