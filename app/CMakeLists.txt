cmake_minimum_required(VERSION 3.29)

project(arenai)

include(AndroidNdkModules)
android_ndk_import_module_native_app_glue()

include(FetchContent)

##############
# ExecuTorch #
##############

FetchContent_Declare(
        executorch
        GIT_REPOSITORY https://github.com/pytorch/executorch.git
        GIT_TAG        "viable/strict"
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        GIT_SUBMODULES_RECURSE TRUE
        SOURCE_DIR     "${CMAKE_CURRENT_BINARY_DIR}/_deps/executorch"
)

set(ET_VENV "${CMAKE_CURRENT_BINARY_DIR}/.venv_et")
execute_process(
        COMMAND bash -lc
        "/usr/bin/python3.12 -m venv \"${ET_VENV}\" \
        && ${ET_VENV}/bin/pip install --upgrade pip pyyaml \
        && ${ET_VENV}/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu"
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _venv_rc
)
if(NOT _venv_rc EQUAL 0)
    message(FATAL_ERROR "Error when creating ExecuTorch python venv")
endif()

set(PYTHON_EXECUTABLE "${ET_VENV}/bin/python" CACHE FILEPATH "")

set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "")
set(EXECUTORCH_BUILD_TESTS OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON CACHE BOOL "")
set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "")
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "")
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "")
set(EXECUTORCH_BUILD_VULKAN OFF CACHE BOOL "")

FetchContent_MakeAvailable(executorch)

#########
# PhyVR #
#########

set(CMAKE_CXX_STANDARD 20)

set(ARENAI_FOLDER "${PROJECT_ROOT}/arenai")
set(ARENAI_BINARY_ROOT_PATH "${CMAKE_CURRENT_BINARY_DIR}/arenai")

add_subdirectory("${ARENAI_FOLDER}/arenai_utils" "${ARENAI_BINARY_ROOT_PATH}/arenai_utils")
add_subdirectory("${ARENAI_FOLDER}/arenai_controller" "${ARENAI_BINARY_ROOT_PATH}/arenai_controller")
add_subdirectory("${ARENAI_FOLDER}/arenai_view" "${ARENAI_BINARY_ROOT_PATH}/arenai_view")
add_subdirectory("${ARENAI_FOLDER}/arenai_model" "${ARENAI_BINARY_ROOT_PATH}/arenai_model")
add_subdirectory("${ARENAI_FOLDER}/arenai_core" "${ARENAI_BINARY_ROOT_PATH}/arenai_core")

include_directories(${ARENAI_UTILS_INCLUDE_DIRS})
include_directories(${ARENAI_CONTROLLER_INCLUDE_DIRS})
include_directories(${ARENAI_VIEW_INCLUDE_DIRS})
include_directories(${ARENAI_MODEL_INCLUDE_DIRS})
include_directories(${ARENAI_CORE_INCLUDE_DIRS})

file(GLOB_RECURSE ARENAI_ANDROID_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/*.cpp")

add_library(arenai SHARED ${ARENAI_ANDROID_SRC})

target_link_libraries(
        arenai
        -Wl,--whole-archive native_app_glue -Wl,--no-whole-archive
        android
        jnigraphics
        arenai_utils
        arenai_view
        arenai_model
        arenai_core
        executorch
        executorch::backends
        executorch::extensions
        executorch::kernels
)
