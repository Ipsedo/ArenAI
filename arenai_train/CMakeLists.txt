cmake_minimum_required(VERSION 3.29)

project(arenai_train_host)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

# Save to png with STB
FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4 # master @ 2024
)
FetchContent_MakeAvailable(stb)

# argparse
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v3.2
)
FetchContent_MakeAvailable(argparse)

# json
FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

# libtorch
if(NOT DEFINED LIBTORCH_PATH)
    set(LIBTORCH_PATH "/opt/libtorch")
endif()
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# SOIL
FetchContent_Declare(
        soil2
        GIT_REPOSITORY https://github.com/SpartanJ/SOIL2.git
        GIT_TAG af13ed898a05fb1dbd4c16306e630f9caf27a6f6 # 1.31 + fixes
)
FetchContent_MakeAvailable(soil2)

# ARENAI modules
set(ARENAI_CPP_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_controller" "${CMAKE_CURRENT_BINARY_DIR}/arenai_controller")
add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_utils" "${CMAKE_CURRENT_BINARY_DIR}/arenai_utils")
add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_view" "${CMAKE_CURRENT_BINARY_DIR}/arenai_view")
add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_model" "${CMAKE_CURRENT_BINARY_DIR}/arenai_model")
add_subdirectory("${ARENAI_CPP_FOLDER}/arenai_core" "${CMAKE_CURRENT_BINARY_DIR}/arenai_core")

set(ARENAI_TRAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE ARENAI_TRAIN_SRC "${ARENAI_TRAIN_PATH}/*.cpp")

# Dependencies shared by the executable and the shared library
set(ARENAI_TRAIN_LINK_PUBLIC_LIBS
        arenai_model
        arenai_view
        arenai_utils
        arenai_core
)

set(ARENAI_TRAIN_LINK_PRIVATE_LIBS
        ${TORCH_LIBRARIES}
        soil2
        argparse
        nlohmann_json::nlohmann_json
)


# Executable
add_executable(arenai_train_exec ${ARENAI_TRAIN_SRC})
target_link_libraries(arenai_train_exec PUBLIC ${ARENAI_TRAIN_LINK_PUBLIC_LIBS})
target_link_libraries(arenai_train_exec PRIVATE ${ARENAI_TRAIN_LINK_PRIVATE_LIBS})
target_include_directories(arenai_train_exec PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${stb_SOURCE_DIR}")
arenai_copy_runtime_dlls(arenai_train_exec TORCH)

# Library
add_library(arenai_train_lib SHARED ${ARENAI_TRAIN_SRC})
target_link_libraries(arenai_train_lib PUBLIC ${ARENAI_TRAIN_LINK_PUBLIC_LIBS})
target_link_libraries(arenai_train_lib PRIVATE ${ARENAI_TRAIN_LINK_PRIVATE_LIBS} "${stb_SOURCE_DIR}")
target_include_directories(arenai_train_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
