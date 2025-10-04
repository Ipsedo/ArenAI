cmake_minimum_required(VERSION 3.29)

project(phyvr)

include(FetchContent)

include(AndroidNdkModules)
android_ndk_import_module_native_app_glue()

##############
# ExecuTorch #
##############

set(ET_SOURCE_DIR "${CMAKE_SOURCE_DIR}/build/executorch")
set(PYTHON_VENV_DIR "${CMAKE_SOURCE_DIR}/build/venv_et")
set(PYTHON_EXECUTABLE "${PYTHON_VENV_DIR}/bin/python3.12")
execute_process(
        COMMAND bash -c
        "python3.12 -m venv ${CMAKE_SOURCE_DIR}/build/venv_et"
)

option(EXECUTORCH_ENABLE_LOGGING "" ON)
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_PYBIND "" OFF)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)
FetchContent_Declare(
        executorch
        GIT_REPOSITORY https://github.com/pytorch/executorch.git
        GIT_TAG        "v0.7.0"
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        GIT_SUBMODULES_RECURSE TRUE
        SOURCE_DIR     "${ET_SOURCE_DIR}"
)
FetchContent_Populate(executorch)
execute_process(
        COMMAND bash -lc
        "source ${PYTHON_VENV_DIR}/bin/activate \
        && cd '${ET_SOURCE_DIR}' \
        && git submodule sync \
        && git submodule update --init --recursive \
        && ./install_requirements.sh \
        && ./install_executorch.sh --clean"
        RESULT_VARIABLE _et_submods_rv
)

add_subdirectory("${ET_SOURCE_DIR}")

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
        extension_module_static
        extension_flat_tensor
        extension_tensor
        optimized_native_cpu_ops_lib
        xnnpack_backend
)
