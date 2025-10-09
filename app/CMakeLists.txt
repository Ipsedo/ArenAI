cmake_minimum_required(VERSION 3.29)

project(phyvr)

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
set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_PYTHON OFF CACHE BOOL "")
set(EXECUTORCH_ENABLE_PYTHON_API OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "")
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "")
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "")
set(EXECUTORCH_BUILD_VULKAN OFF CACHE BOOL "")

FetchContent_MakeAvailable(executorch)

#########
# PhyVR #
#########

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
        executorch
        executorch::backends
        executorch::extensions
        executorch::kernels
)
