cmake_minimum_required(VERSION 3.29)

project(arenai_view)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

##### GLM #####
find_package(glm REQUIRED)

##### Vulkan #####
# Headers + loader: vcpkg (vulkan-headers/vulkan-loader) on Windows, distro
# packages on Linux. The runtime driver comes from the system ICDs.
find_package(Vulkan REQUIRED)

# glslang turns shaders_vk/*.glsl into SPIR-V at build time (host tool built
# from source: no Vulkan SDK requirement). Static, so it MUST be fetched
# before BUILD_SHARED_LIBS goes ON for RmlUi below.
include("${CMAKE_SOURCE_DIR}/cmake/ArenaiCompileShaders.cmake")
arenai_fetch_glslang()

##### VulkanMemoryAllocator (header-only) #####
FetchContent_Declare(
        vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(vma)

##### RmlUi (UI library) #####
# Built shared (this directory's scope only) so the code links into our shared
# libraries without a PIC surprise. Needs FreeType on the system for fonts.
set(BUILD_SHARED_LIBS ON)
set(RMLUI_SAMPLES OFF)
FetchContent_Declare(
        rmlui
        GIT_REPOSITORY https://github.com/mikke89/RmlUi.git
        GIT_TAG 6.2
)
FetchContent_MakeAvailable(rmlui)

# Vulkan shaders (shaders/*.glsl): compiled to SPIR-V by glslang at build
# time and embedded into the library (EMBEDDED_SPIRV map in shaders_spirv.h,
# see cmake/ArenaiCompileShaders.cmake). The built executables need no shader
# file at runtime, and shader errors break the build instead of the first frame.
set(ARENAI_VIEW_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ARENAI_VIEW_GENERATED_SOURCES "")
arenai_compile_shaders(
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "${ARENAI_VIEW_GENERATED_DIR}"
        "${ARENAI_VIEW_GENERATED_DIR}/shaders_spirv.h"
        ARENAI_VIEW_GENERATED_SOURCES)

# ARENAI VIEW
file(GLOB_RECURSE ARENAI_VIEW_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_view SHARED ${ARENAI_VIEW_SRC} ${ARENAI_VIEW_GENERATED_SOURCES})
target_include_directories(arenai_view PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(arenai_view PRIVATE "${ARENAI_VIEW_GENERATED_DIR}")

# glm::glm carries its own include directories, so no manual include path is needed.
target_link_libraries(
        arenai_view
        PUBLIC
        arenai_utils
        arenai_controller
        glm::glm
)

find_package(glfw3 REQUIRED)
target_link_libraries(arenai_view PRIVATE glfw)

# Vulkan stays PRIVATE like GL before it: no Vulkan type crosses the public API.
target_link_libraries(arenai_view PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)

# PRIVATE like GL/GLFW: the public headers only forward-declare
# Rml::RenderInterface, the library itself never leaks past arenai_view.
target_link_libraries(arenai_view PRIVATE RmlUi::Core)

