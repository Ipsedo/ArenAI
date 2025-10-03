cmake_minimum_required(VERSION 3.11)

project(phyvr_train_host)

set(CMAKE_CXX_STANDARD 20)

# libtorch
set(LIBTORCH_PATH "/opt/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

# PHYVR
set(PHYVR_CPP_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/cpp")

add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_controller" "${CMAKE_CURRENT_BINARY_DIR}/phyvr_controller")
add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_utils" "${CMAKE_CURRENT_BINARY_DIR}/phyvr_utils")
add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_view" "${CMAKE_CURRENT_BINARY_DIR}/phyvr_view")
add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_model" "${CMAKE_CURRENT_BINARY_DIR}/phyvr_model")
add_subdirectory("${PHYVR_CPP_FOLDER}/phyvr_core" "${CMAKE_CURRENT_BINARY_DIR}/phyvr_core")

include_directories(${PHYVR_CONTROLLER_INCLUDE_DIRS})
include_directories(${PHYVR_UTILS_INCLUDE_DIRS})
include_directories(${PHYVR_VIEW_INCLUDE_DIRS})
include_directories(${PHYVR_MODEL_INCLUDE_DIRS})
include_directories(${PHYVR_CORE_INCLUDE_DIRS})

set(PHYVR_TRAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE PHYVR_TRAIN_SRC "${PHYVR_TRAIN_PATH}/*.cpp")

add_executable(phyvr_train_host ${PHYVR_TRAIN_SRC}
        src/vision.h
        src/agent.cpp
)

target_link_libraries(
        phyvr_train_host
        phyvr_model
        phyvr_view
        phyvr_utils
        phyvr_core
        ${TORCH_LIBRARIES}
)
