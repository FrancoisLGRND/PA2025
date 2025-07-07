cmake_minimum_required(VERSION 3.30)
project(PA2025)

set(CMAKE_CXX_STANDARD 20)

add_library(linear_model SHARED classification_lineaire.cpp)
