cmake_minimum_required(VERSION 3.11)

project(arenai_train_host)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

# Save to png with STB
FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
)

FetchContent_Populate(stb)
include_directories(${stb_SOURCE_DIR})

# libtorch
set(LIBTORCH_PATH "/opt/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

# SOIL
find_package(soil2 REQUIRED)
include_directories(${SOIL2_INCLUDE_DIR})

# ARENAI
set(ARENAI_CPP_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_controller" "${CMAKE_CURRENT_BINARY_DIR}/arenai_controller")
add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_utils" "${CMAKE_CURRENT_BINARY_DIR}/arenai_utils")
add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_view" "${CMAKE_CURRENT_BINARY_DIR}/arenai_view")
add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_model" "${CMAKE_CURRENT_BINARY_DIR}/arenai_model")
add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_core" "${CMAKE_CURRENT_BINARY_DIR}/arenai_core")

include_directories(${ARENAI_CONTROLLER_INCLUDE_DIRS})
include_directories(${ARENAI_UTILS_INCLUDE_DIRS})
include_directories(${ARENAI_VIEW_INCLUDE_DIRS})
include_directories(${ARENAI_MODEL_INCLUDE_DIRS})
include_directories(${ARENAI_CORE_INCLUDE_DIRS})

set(ARENAI_TRAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE ARENAI_TRAIN_SRC "${ARENAI_TRAIN_PATH}/*.cpp")

add_executable(arenai_train_host ${ARENAI_TRAIN_SRC})

target_link_libraries(
        arenai_train_host
        arenai_model
        arenai_view
        arenai_utils
        arenai_core
        ${TORCH_LIBRARIES}
        soil2
)
