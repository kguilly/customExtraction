# README
### This project aims to extract WRF GRIB1 and GRIB2 files for selected parameters in selected locations across the United States using a C/C++ implementation of the ECCODES library
### **Note:** This README is written for debian linux systems. ECCODES does not directly support windows systems, however community support can be found [here](https://github.com/moonpyk/eccodes-build-windows)
#### The opening part of this README is based on a [tutorial for installing ECCODES](https://gist.github.com/MHBalsmeier/a01ad4e07ecf467c90fad2ac7719844a)
1. Prepare your system
   - From the command line, install the necessary prerequisite libraries with the following comand
      ```
      sudo apt-get install libnetcdff-dev libopenjp2-7-dev gfortran make unzip git cmake wget
      ```
   - If not already, create a directory to store source builds
      ```
      cd ; mkdir source_builds ; cd source_builds ; mkdir eccodes ; cd eccodes
      ```
2. Installing the source code
   - Download the source code through either navigating to the [ECCODES downloads page](https://confluence.ecmwf.int/display/ECC/ecCodes+installation) and choosing the a select version of ECCODES or through running the following command while inside the eccodes folder to download version 2.27.0
      ```
      wget https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.27.0-Source.tar.gz
      ```
   - Extract the code in the eccodes directory
      ```
      tar -xzf eccodes-2.27.0-Source.tar.gz
      ```
3. Building
   - Make a folder to build from
      ```
      mkdir build ; cd build
      ```
   - Choose a folder to install into. For the purposes of this README, /usr/src/ will be used
      ```
      sudo mkdir /usr/src/eccodes
      cmake -DCMAKE_INSTALL_PREFIX=/usr/src/eccodes -DENABLE_JPG=ON -DENABLE_ECCODES_THREADS=ON -DENABLE_FORTRAN=OFF ../eccodes-2.27.0-Source
      make
      ctest
      ```
   - Ensure all tests are passed before installing with:
      ```
      sudo make install
      ```
   - Run the following command to use ECCODES functions from the command line (not needed for running code in this repository)
      ```
      sudo cp -r /usr/src/eccodes/bin/* /usr/bin
      ```
4. Setting Environment variables
   - If installing ECCODES for the first time. Run the following commands in order to properly link the libraries
      ```
      echo 'export ECCODES_DIR=/usr/src/eccodes' >> ~/.bashrc
      echo 'export ECCODES_DEFINITION_PATH=/usr/src/eccodes/share/eccodes/definitions' >> ~/.bashrc
      source ~/.bashrc
      ```
   - Copy the shared libraries and header files into their standard locations with the following command
      ```
      sudo cp $ECCODES_DIR/lib/libeccodes.so /usr/lib
      sudo cp /usr/src/eccodes/include/* /usr/include
      ```

