cmake_minimum_required(VERSION 3.11)

project(phyvr_controller)

include(FetchContent)

##### PhyVR #####
set(CMAKE_CXX_STANDARD 20)

set(PHYVR_CONTROLLER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
