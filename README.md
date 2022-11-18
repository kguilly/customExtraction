# README
### This project aims to extract WRF GRIB1 and GRIB2 files for selected parameters in selected locations across the United States using a C/C++ implementation of the ECCODES library
### **Note:** This README is written for debian linux systems. ECCODES does not directly support windows systems, however community support can be found [here](https://github.com/moonpyk/eccodes-build-windows)
1. Download latest version of [ECCODES](https://confluence.ecmwf.int/display/ECC)
   - Scroll down and click the blue button that says "Download the latest tarball"
   - In your command prompt, create a new directory to place ECCODES in
        ```
        $mkdir eccodes ; cd eccodes
        ```
   - Using either the gui or the command prompt, extract the tarball into the directory.
        ```
        $tar xvf eccodes-x.y.z-Source.tar.gz ; mv eccodes-x.y.z-Source ../eccodes
        ```
   - In the eccodes directory, make a new folder to build into
        ```
        $cd eccodes
        $mkdir build ; cd build
        ```
   - Install CMake. This can either be done by installing the [gui](https://cmake.org/install/) or through the command prompt.
        ```
        $sudo apt install cmake
        ```
2. Clone / Download zip file of the [customExtraction repository]()
