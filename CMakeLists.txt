cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(raytrace LANGUAGES CXX CUDA)

add_executable(raytrace main.cu)

set_target_properties(raytrace PROPERTIES
    CUDA_ARCHITECTURES 86
)

set(CMAKE_BUILD_TYPE Release)