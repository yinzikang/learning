cmake_minimum_required(VERSION 3.5)
project(jk_kdl_test)

find_package(orocos_kdl REQUIRED)
include_directories(${orocos_kdl_INCLUDE_DIRS})
message(${orocos_kdl_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} ${orocos_kdl_LIBRARIES})