cmake_minimum_required(VERSION 3.23)
project(myFiles)

set(CMAKE_CXX_STANDARD 14)

add_executable(myFiles
        extractWRFData1.cpp
        extractWRFData2.cpp
        extractWRFData3.cpp
        get_data_offEx.cpp
        grib_print_data.cpp
        importGribTest.cpp
        test1.cpp
        threadedExtractWRFData1.cpp
        threadingTest.cpp)
