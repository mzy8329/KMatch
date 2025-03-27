cmake_minimum_required(VERSION 3.12)
project(KMatch)

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(KMatch KMatch.cpp)
target_include_directories(KMatch PRIVATE ${EIGEN3_INCLUDE_DIRS})