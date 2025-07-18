cmake_minimum_required(VERSION 3.30)
project(PA2025)

set(CMAKE_CXX_STANDARD 20)

add_library(linear_model SHARED classification_lineaire.cpp)
add_library(mlp SHARED mlp.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mlp PUBLIC OpenMP::OpenMP_CXX)
endif()