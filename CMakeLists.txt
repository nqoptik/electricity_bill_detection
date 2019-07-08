cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(electricity_bill)

find_package(OpenCV REQUIRED)

add_executable(electricity_bill src/electricity_bill.cpp)
target_link_libraries(electricity_bill ${OpenCV_LIBS})
