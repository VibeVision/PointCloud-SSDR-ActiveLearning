

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(LIBGEOF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")

##############################
### Find required packages ###
##############################
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

find_package(PythonLibs)
find_package(PythonInterp)
find_package(NumPy 1.5 REQUIRED)

find_package(Boost 1.65.0 COMPONENTS graph REQUIRED) #system filesystem thread serialization
if (${Boost_MINOR_VERSION} LESS 67 )
    find_package(Boost 1.65.0 COMPONENTS numpy