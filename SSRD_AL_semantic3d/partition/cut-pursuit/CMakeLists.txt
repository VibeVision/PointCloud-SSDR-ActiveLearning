# top-level CMake configuration file

cmake_minimum_required(VERSION 3.5)

project(CUT_PURSUIT_SEG)


#------------------------------------------------------------------------------
# internal cmake settings
#------------------------------------------------------------------------------
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_MACO