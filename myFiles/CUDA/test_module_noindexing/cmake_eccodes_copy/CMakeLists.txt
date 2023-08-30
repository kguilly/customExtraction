# declare min version
cmake_minimum_required(VERSION 3.16.3)

# NAME the project
project(ECCODES_COPIED)
add_executable(${PROJECT_NAME} min_decompression.cpp)
add_library(eccodes ./src/*.c)