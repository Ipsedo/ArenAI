cmake_minimum_required(VERSION 3.29)

project(arenai_view)

include(FetchContent)

##### GLM #####
if (ANDROID)
    FetchContent_Declare(glm URL https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip)
    FetchContent_MakeAvailable(glm)
    include_directories("${glm_SOURCE_DIR}")
else ()
    find_package(glm REQUIRED)
    include_directories(${GLM_INCLUDE_DIRS})
endif ()

# ARENAI VIEW
set(CMAKE_CXX_STANDARD 20)

set(ARENAI_VIEW_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

include_directories(${ARENAI_CONTROLLER_INCLUDE_DIRS})
include_directories(${ARENAI_UTILS_INCLUDE_DIRS})
include_directories(${ARENAI_VIEW_INCLUDE_DIRS})

file(GLOB_RECURSE ARENAI_VIEW_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(arenai_view SHARED ${ARENAI_VIEW_SRC})
set_target_properties(arenai_view PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_link_libraries(
        arenai_view
        arenai_utils
        glm::glm
)

if (ANDROID)
    target_link_libraries(arenai_view EGL GLESv3)
    target_include_directories(arenai_view PUBLIC "${glm_SOURCE_DIR}")
elseif (WIN32)
    # Mesa3D SDK (headers + import libs) from mesa-dist-win "devel" package,
    # fetched by arenai_desktop/install_dependencies.ps1 into libs/mesa-devel.
    if (NOT MESA_SDK_DIR)
        set(MESA_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../arenai_desktop/libs/mesa-devel")
    endif ()
    find_path(MESA_INCLUDE_DIR GLES3/gl3.h HINTS "${MESA_SDK_DIR}/include" REQUIRED)
    find_library(MESA_EGL_LIB libEGL HINTS "${MESA_SDK_DIR}/lib/x64" REQUIRED)
    find_library(MESA_GLES2_LIB libGLESv2 HINTS "${MESA_SDK_DIR}/lib/x64" REQUIRED)
    target_link_libraries(arenai_view "${MESA_EGL_LIB}" "${MESA_GLES2_LIB}")
    target_include_directories(arenai_view PUBLIC "${GLM_INCLUDE_DIRS}" "${MESA_INCLUDE_DIR}")
else ()
    target_link_libraries(arenai_view EGL GLESv2 dl)
    target_include_directories(arenai_view PUBLIC "${GLM_INCLUDE_DIRS}")
endif ()
