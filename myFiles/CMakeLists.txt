cmake_minimum_required(VERSION 3.23)
project(myFiles)

set(CMAKE_CXX_STANDARD 14)

add_executable(myFiles threadedExtractWRFData1.cpp)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(ecbuild 3.7 REQUIRED HINT /home/kalebg/Desktop/School/Y4S1/REU/newEccodes_10_13/eccodes-2.27.0-Source/cmake/)
target_link_libraries(myFiles Threads::Threads)
target_link_libraries(myFiles eccodes)

