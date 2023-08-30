# from the gpt suggestion
# Add the library
cmake_minimum_required(VERSION 3.16.3)
project(EC_COPY)
add_executable(${PROJECT_NAME} minimum_get_data.c)
add_library(eccodes SHARED IMPORTED)

# Set the path to the library file
set_target_properties(eccodes PROPERTIES IMPORTED_LOCATION "/usr/src/eccodes/lib/libeccodes.so")

# Set the path to the library's header files
target_include_directories(${PROJECT_NAME} PUBLIC "/usr/src/eccodes/include")

# Link the library to your target
target_link_libraries(${PROJECT_NAME} PUBLIC eccodes)
