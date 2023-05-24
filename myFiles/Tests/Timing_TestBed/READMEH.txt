This testbed will be used to time the python execution vs C exeuction.

The test will encompass:

    - 7 grib parameters:
        - 2 m temperature
        - 2 m relative humidity
        - downward shortwave radiation flux
        - Wind Gust
        - u component of wind (level 1000)
        - v component of wind (level 1000)
        - precipitation (use 1 hour prediction)
    
    - One state: Louisiana
        - separated county by county, each county divided into 2km by 2km grids 

    - One month: January 2022

Profiling stats:
    C:
        - one file
            time: 
            mem: 

        - one day
            time: 
            mem: 229.2 MB

        - one month
            time: 
            mem: 


    Python:
        -one file
            time:
            mem:

        - one day
            time:
            mem:

        - one month 
            time:
            mem:


The information in the GRIB2 files will only be decompressed, the data will not be sorted
or further processed 

Modifications:
    * do not pass fips flag, i've already compiled the wrfoutput.csv file to include
      all louisiana counties. 
    C:
        - add precip_flag and code to extract precipitation files
        - compile: g++ -std=c++17 -w c_extraction_timingTest.cpp -leccodes -lpthread
            - run: ./a.out --param 9 36 37 71 75 123  --precip_flag --begin_date 20200101 --end_date 20200201
        - modify the output files
            - accomodate the header to include the precipitation
            - monthly file for each state (one file)
    Python: 
        - add lines to time code
        - remove code that further processes the files, only 
          include code that decompresses the file. 
