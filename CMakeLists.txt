cmake_minimum_required(VERSION 3.5)

project(electricity_bill_detection)

## Compile as C++14, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Compile with the highest warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

## System dependencies
find_package(OpenCV REQUIRED)
if(NOT ${OpenCV_VERSION} STRGREATER "2.4")
    message(FATAL_ERROR "OpenCV_VERSION has to > 2.4")
endif()

## Specify additional locations of header files
include_directories(${catkin_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

## Declare a C++ executable
add_executable(electricity_bill_detection src/electricity_bill_detection.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(electricity_bill_detection ${OpenCV_LIBS})
