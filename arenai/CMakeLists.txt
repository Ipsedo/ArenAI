cmake_minimum_required(VERSION 3.11)

project(arenai)

# Train main & lib
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_train")

# Desktop main (game)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_desktop")

# Tests
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/arenai_view/tests")
