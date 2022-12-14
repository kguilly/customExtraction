/*
This file will extract a given list of parameters from a given list of lattitudes
and longitudes from WRF GRIB2 files

Its functionality is based on extractWRFData.py by Dr. Purushottam Sigdel as well as
grib_keys_iterator.c and grib_get_data.c found in the examples of the eccodes library.


Compile:
g++ -Wall extractWRFData.cpp -leccodes 

*/

#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include "eccodes.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector> 
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include <time.h>
#include <cassert>
#include "semaphore.h"
#define MAX_VAL_LEN 1024
using namespace std;

/* Timing variables */
struct timespec startExtract; 
struct timespec endExtract;
struct timespec startMatching;
struct timespec endMatching;
struct timespec startTotal;
struct timespec endTotal;
double extractTime;
double matchTime;
double totalTime;

vector<int> beginDay = {2019, 1, 2}; // arrays for the begin days and end days. END DAY IS NOT INCLUSIVE.  
                                    // when passing a single day, pass the day after beginDay for endDay
                                    // FORMAT: {yyyy, mm, dd}
vector<int> endDay = {2019, 1, 3};

vector<int> arrHourRange = {0,23}; // array for the range of hours one would like to extract from
                                 // FORMAT: {hh, hh} where the first hour is the lower hour, second is the higher
                                 // accepts hours from 0 to 23 (IS INCLUSIVE)

int intHourRange; 

string filePath = "/home/kalebg/Desktop/School/Y4S1/REU/extraction/UtilityTools/extractTools/data/";  // path to "data" folder. File expects structure to be: 
                                        // .../data/<year>/<yyyyMMdd>/hrrr.<yyyyMMdd>.<hh>.00.grib2
                                        // for every hour of every day included. be sure to include '/' at end
string writePath = "/home/kalebg/Desktop/WRFData/";   //path to write extracted data to

// Structure for holding the selected station data. Default will be the 5 included in the acadmeic
// paper: "Regional Weather Forecasting via Neural Networks with Near-Surface Observational and Atmospheric Numerical Data."
struct Station{
    string name;
    float lat;
    float lon;
    double *values; // holds the values of the parameters. Index of this array will 
                    // correspond to index if the Parameter array. This will be a single hour's data
    map<string, vector<double>> dataMap; // this will hold the output data. structured as {"yyyymmddHH" : [param1, param2, ...]}}

    int closestPoint; // index in the grib file of the point closest to the station's lats and lons
    
    // NOT IMPLEMENTED YET
    // the next values will store the index of the parameters around the station that are returned by the GRIB file
    int topL[2]; // index of the lat and lons of the point up and to the left. 
    int topR[2]; 
    int botL[2];
    int botR[2];
};

struct Parameter{
    int layer; // order that the parameter is extracted in decompression. parameter number in ACM paper
               // layer is NOT reliable. Some parameters are repeated in decompression. Parameters with a height attached to them
               // are not shown in the "name" from the key iterator
    int paramId; // given from codes_keys_iterator object
    string shortName; //given
    string units; //given
    string name; // given
};


Station *stationArr; 
Parameter *objparamArr; // array of the WRF parameter names. Default is the 30 in 
                          // the academic paper "Regional Weather Forecasting via Neural
                          // Networks with Near-Surface Observational and Atmospheric Numerical 
                          // Data." Names are changed to fit the exact names
bool blnParamArr[149]; // this will be used to quickly index whether a parameter needs to be 
                        // extracted or not. Putting 149 spaces for 148 parameters because the
                        // layers of the parameters start at 1 rather than 0

int numStations, numParams;

// 24 hours in a day. Use this to append to the file name and get each hour for each file
string *hours;

// function to handle arguments passed. Will either build the Station array and/or paramter array
// based off of the arguments passed or will build them with default values. Will also construct
// the hour array based on passed beginning and end hours
void handleInput(int, char**);

/* function to convert the lats and lons from the passed representation to the 
    way they are represented in the grib file. ONLY WORKS for coordinates in USA*/
void convertLatLons();

// function to check the begin day vs the end day to see if a valid range has been passed
bool checkDateRange(vector<int>, vector<int>);

// function to get the next day after the day passed
vector<int> getNextDay(vector<int>);

/* function to format the date as a vector of strings
   returns vector<string> date = {yyyy, mm, dd, yyyymmdd}*/
vector<string> formatDay(vector<int>);

/* function to check if a given directory exists*/
bool dirExists(string filePath);

/* function to read the data from a passed grib file
    After reading the data, the *values array for each station
    will be filled with the select paramters for a given hour */
void readData(FILE*, string, string);

/* function to map the data in the station's values array to the station's map */
void mapData(string, string);

/*function to write the data in the station maps to a .csv file*/
void writeData(vector<string>, string);

int main(int argc, char*argv[]){

    clock_gettime(CLOCK_MONOTONIC, &startTotal);

    handleInput(argc, argv);

    convertLatLons();

    // validate the dates passed
    bool boolDateRange = checkDateRange(beginDay, endDay);
    if (boolDateRange == false){
        fprintf(stderr, "Invalid Date Range\n");
        exit(1);
    }
    vector<int> intcurrentDay = beginDay;

    // for each day
    while(checkDateRange(intcurrentDay, endDay)){
        vector<string> strCurrentDay = formatDay(intcurrentDay);

        // concatenate the file path
        string filePath1 = filePath + strCurrentDay.at(0) + "/" + strCurrentDay.at(3) + "/";
        // check if the file path exists
        if(dirExists(filePath1) == false){
            fprintf(stderr, "Error: could not find directory %s", filePath1.c_str());
            exit(1);
        }

        FILE* f[intHourRange]; // use to open the file for each hour
        
        for(int i=0;i<intHourRange;i++){ // for each hour, thread the file and filename
            f[i] = NULL;
            string hour = hours[i];
            string fileName = "hrrr."+strCurrentDay.at(3)+"."+hour+".00.grib2";
            string filePath2 = filePath1 + fileName;
            
            readData(*f, fileName, filePath2);
            mapData(strCurrentDay.at(3), hour);
            writeData(strCurrentDay, hour);

        }     
        intcurrentDay = getNextDay(intcurrentDay);
    }


    // print out all the elements in all the station's data maps
    for (int i=0; i<numStations;i++){
        Station station = stationArr[i];
        cout << "\n\nSTATION: " << station.name << endl;
        for(auto itr = station.dataMap.begin(); itr != station.dataMap.end(); ++itr){
            cout << itr->first << '\t';
            for (int i = 0; i<itr->second.size(); i++){
                 cout << itr->second.at(i) << " ";
            }
            cout << endl;
        }
    }

    for(int i =0; i<numStations;i++){
        // delete each station's value array
        delete [] stationArr[i].values;
    }
    delete [] objparamArr;
    delete [] stationArr;
    std::free(hours);

    clock_gettime(CLOCK_MONOTONIC, &endTotal);
    totalTime = (endTotal.tv_sec - startTotal.tv_sec) * 1000.0;
    totalTime+= (endTotal.tv_nsec - startTotal.tv_nsec) / 1000000.0;
    printf("\n\nRuntime in ms:: %f\n", totalTime);
    printf("Extract Time in ms:: %f\n", extractTime);
    printf("Time to find index: %f\n\n", matchTime);

    return 0;
 }

 void handleInput(int argc, char* argv[]){
    
    // NOT FINISHED. DO NOT PASS ARGS
    if(argc > 1){
        // check to see if the correct arguments have been passed to fill the parameter and/or 
        // station arrays
    }
    else{
        // build the default staion array and parameter array
        // potential improvement: for each month, run the file and make new parameter array based off
        //                        of what the file returns
        numParams = 148;
        numStations = 5;
        
        
        Parameter refc1, ATT2, veril3, vis4, refd5, refd6, refd7, gust8, u9, v10, u11, v12, gh13,
                  t14, dpt15, u16, v17, gh18, t19, dpt20, u21, v22, gh23, t24, dpt25, u26, 
                  v27, t28, dpt29, u30, v31, t32, dpt33, u34, v35, ATT37, wz38, msla39, gh40,
                  ATT41, refd42, ATT43, ATT44, ATT45, ATT46, ATT47, ATT48, vo49, vo50, hail51,
                  hail52, ATT53, ltng54, u55, v56, sp57, orog58, t59, asnow60, mstav61,cnwat62,
                  sdwe63, snowc64, sde65, twot66, pt67, twosh68, twod69, twor70, tenu71, tenv72,
                  tensi73, ATT74, ATT75, cpofp76, prate77, tp78, sdwe79, ATT80, frzr81, ssrun82,
                  bgrun83, csnow84, cicep85, cfrzr86, crain87, sr88, fricv89, shtfl90, lhtfl91,
                  gflux92, vgtyp93, lftx94, cape95, cin96, pwat97, lcc98, mcc99, hcc100, tcc101,
                  pres102, gh103, gh104, pres105, gh106, ulwrf107, dswrf108, dlwrf109, uswrf110,
                  ulwrf111, vbdsf112, vddsf113, uswrf114, hlcy115, hlcy116, ustm117, vstm118,
                  vucsh119, vvcsh120, vucsh121, vvcsh122, gh123, r124, pres125, gh126, r127,
                  pres128, gh129, gh130, fourlftx131, cape132, cin133, hpbl134, gh135, cape136,
                  cin137, cape138, cin139, gh140, plpl141, ATT142, lsm143, ci144, sbt123145,
                  sbt124146, sbt113147, sbt114148;

        refc1.layer = 1, refc1.name = "Maximum/Composite Radar Reflectivity", refc1.units = ;
        ATT2.layer = 2, ATT2.name = "unkn", ATT2.units = ;
        veril3.layer = 3, veril3.name = "Vertically-Integrated Liquid", veril3.units = ;
        vis4.layer = 4, vis4.name = "Visibility", vis4.paramId = 3020, vis4.shortName = "vis", vis4.units = "m";
        refd5.layer = 5, refd5.name = "Derived Radar Reflectivity", refd5.units = ;
        refd6.layer = 6, refd6.name = "Derived Radar Reflectivity", refd6.units = ;
        refd7.layer = 7, refd7.name = "Derived Radar Reflectivity", refd7.units = ;
        gust8.layer = 8, gust8.name = "Wind speed (gust)", gust8.paramId = 260065, gust8.shortName = "gust", gust8.units = "m s**-1";
        u9.layer = 8, u9.name = "U component of wind", u9.units = ;
        v10.layer = 10, v10.name = "V component of wind", v10.units = ;
        u11.layer = 11, u11.name = "U component of wind", u11.units = ;
        v12.layer = 12, v12.name = "V component of wind", v12.units = ;
        gh13.layer = 13, gh13.name = "Geopotential Height", gh13.units = ;
        t14.layer = 14, t14.name = "Temperature", t14.units = ;
        dpt15.layer = 15, dpt15.name = "Dew Point Temperature", dpt15.units = ;
        u16.layer = 16, u16.name = "U component of wind", u16.units = ;
        v17.layer = 17, v17.name = "V component of wind", v17.units = ;
        gh18.layer = 18, gh18.name = "Geopotential Height", gh18.units = "gpm";
        t19.layer = 19, t19.name = "Temperature", t19.units = ;
        dpt20.layer = 20, dpt20.name = "Dew Point Temperature", dpt20.units = ;
        u21.layer = 21, u21.name = "U component of wind", u21.units = ;
        v22.layer = 22, v22.name = "V component of wind", v22.units = ;
        gh23.layer = 23, gh23.name = "Geopotential Height", gh23.units = ;
        t24.layer = 24, t24.name = "Temperature", t24.units = ;
        dpt25.layer = 25, dpt25.name = "Dew Point Temperature", dpt25.units = ;
        u26.layer = 26, u26.name = "U component of wind", u26.units = ;
        v27.layer = 27, v27.name = "V component of wind", v27.units = ;
        t28.layer = 28, t28.name = "Temperature", temp.units = "K";
        dpt29.layer = 29, dpt29.name = "Dew Point Temperature", dpt29.units = ;
        u30.layer = 30, u30.name = "U component of wind", u30.units = ;
        v31.layer = 31, v31.name = "V component of wind", v31.units = ;
        t32.layer = 32, t32.name = "Temperature", t32.units = ;
        dpt33.layer = 33, dpt33.name = "Dew Point Temperature", dpt33.units = ;
        u34.layer = 34, u34.name = "U component of wind", u34.units = ;
        v35.layer = 35, v35.name = "V component of wind", v35.units = ;
        ATT37.layer = 36, ATT37.name = "unkn", ATT37.units = ;
        wz38.layer = 38, wz38.name = "Geometric vertical velocity", wz38.units = ;
        msla39.layer = 39, msla39.name = "MSLP (MAPS System Reduction)", msla39.units = "Pa";
        gh40.layer = 40, gh40.name = "Geopotential height", gh40.units = "gpm";
        ATT41.layer = 41, ATT41.name = "unkn", ATT41.units = ;
        refd42.layer = 42, refd42.name = "Derived Radar Reflectivity", refd42.units = ;
        ATT43.layer = 43, ATT43.name = "unkn", ATT43.units = ;
        ATT44.layer = 44, ATT44.name = "unkn", ATT44.units = ;
        ATT45.layer = 45, ATT45.name = "unkn", ATT45.units = ;
        ATT46.layer = 46, ATT46.name = "unkn", ATT46.units = ;
        ATT47.layer = 47, ATT47.name = "unkn", ATT47.units = ;
        ATT48.layer = 48, ATT48.name = "unkn", ATT48.units = ;
        vo49.layer = 49, vo49.name = "Vorticity (relative)", vo49.units = ;
        vo50.layer = 50, vo50.name = "Vorticity (relative)", vo50.units = ;
        hail51.layer = 51, hail51.name = "Hail", hail51.units = ;
        hail52.layer = 52, hail52.name = "Hail", hail52.units = ;
        ATT53.layer = 53, ATT53.name = "unkn", ATT53.units = ;
        ltng54.layer = 54, ltng54.name = "Lightning", ltng54.units = ;
        u55.layer = 55, u55.name = "U component of wind", u55.units = ;
        v56.layer = 56, v56.name = "V component of wind", v56.units = ;
        sp57.layer = 57, sp57.name = "Surface pressure", sp57.paramId = 134, sp57.shortName = "sp", sp57.units = "Pa";
        orog58.layer = 58, orog58.name = "Orography", orog58.units = ;
        t59.layer = 59, t59.name = "Temperature", t59.paramId = 130, t59.shortName = "t", t59.units = "K";
        asnow60.layer = 60, asnow60.name = "Total Snowfall", asnow60.units = ;
        mstav61.layer = 61, mstav61.name = "Moisture availibility", mstav61.paramId = 260187, mstav61.shortName = "mstav", mstav61.units = "%";
        cnwat62.layer = 62, cnwat62.name = "Plant canopy surface water", cnwat62.units = ;
        sdwe63.layer = 63, sdwe63.name = "Water equivalent of accumulated snow depth (deprecated)", sdwe63.units = ;
        snowc64.layer = 64, snowc64.name = "Snow Cover", snowc64.units = ;
        sde65.layer = 65, sde65.name = "Snow Depth", sde65.units = ;
        twot66.layer = 66, twot66.name = "2 metre temperature", twot66.paramId = 167, twot66.shortName = "2t", twot66.units = "K"; 
        pt67.layer = 67, pt67.name = "Potential temperature", pt67.paramId = 3, pt67.shortName = "pt", pt67.units = "K"; 
        twosh68.layer = 68, twosh68.name = "2 metre specific humidity", twosh68.paramId = 174096, twosh68.shortName = "2sh", twosh68.units = "kg kg**-1";
        twod69.layer = 69, twod69.name = "2 metre dewpoint temperature", twod69.paramId = 168, twod69.shortName = "2d", twod69.units = "K";
        twor70.layer = 70, twor70.name = "2 metre relative humidity", twor70.paramId = 260242, twor70.shortName = "2r", twor70.units = "%";
        tenu71.layer = 71, tenu71.name = "10 metre U wind component", tenu71.units = ;
        tenv72.layer = 72, tenv72.name = "10 metre V wind component", tenv72.units = ;
        tensi73.layer = 73, tensi73.name = "10 metre wind speed", tensi73.paramId = 207, tensi73.shortName = "10si", tensi73.units = "m s**-1";
        ATT74.layer = 74, ATT74.name = "unkn", ATT74.units = ;
        ATT75.layer = 75, ATT75.name = "unkn", ATT75.units = ;
        cpofp76.layer = 76, cpofp76.name = "Percent frozen precipitation", cpofp76.units = ;
        prate77.layer = 77, prate77.name = "Precipitation rate", prate77.units = ;
        tp78.layer = 78, tp78.name = "Total Precipitation", tp78.units = ;
        sdwe79.layer = 79, sdwe79.name = "Water equivalent of accumulated snow depth (deprecated)", sdwe79.units = ;
        ATT80.layer = 80, ATT80.name = "unkn", ATT80.units = ;
        frzr81.layer = 81, frzr81.name = "Freezing Rain", frzr81.units = ;
        ssrun82.layer = 82, ssrun82.name = "Storm surface runoff", ssrun82.units = ;
        bgrun83.layer = 83, bgrun83.name = "Baseflow-groundwater runoff", bgrun83.units = ;
        csnow84.layer = 84, csnow84.name = "Categorical Snow", csnow84.units = ;
        cicep85.layer = 85, cicep85.name = "Categorical ice pellets", cicep85.units = ;
        cfrzr86.layer = 86, cfrzr86.name = "Categorical freezing rain", cfrzr86.units = ;
        crain87.layer = 87, crain87.name = "Categorical Rain", crain87.units = ;
        sr88.layer = 88, sr88.name = "Surface roughness", sr88.units = ;
        fricv89.layer = 89, fricv89.name = "Frictional velocity", fricv89.paramId = 260073, fricv89.shortName = "fricv", fricv89.units = "m s**-1";
        shtfl90.layer = 90, shtfl90.name = "Sensible heat net flux", shtfl90.paramId = 260003, shtfl90.shortName = "shtfl", shtfl90.units = "W m**-2";
        lhtfl91.layer = 91, lhtfl91.name = "Latent heat net flux", lhtfl91.paramId = 260002, lhtfl91.shortName = "lhtfl", lhtfl91.units = "W m**-2";
        gflux92.layer = 92, gflux92.name = "Ground Heat Flux", gflux92.units = ;
        vgtyp93.layer = 93, vgtyp93.name = "Vegetation Type", vgtyp93.units = ;
        lftx94.layer = 94, lftx94.name = "Surface lifted index", lftx94.paramId = 260127, lftx94.shortName = "lftx", lftx94.units = "K";
        cape95.layer = 95, cape95.name = "Convective available potential energy", cape95.units = ;
        cin96.layer = 96, cin96.name = "Convective inhibition", cin96.units = ;
        pwat97.layer = 97, pwat97.name = "Precipitable water", pwat97.units = ;
        lcc98.layer = 98, lcc98.name = "Low cloud cover", lcc98.paramId = 3073, lcc98.shortName = "lcc98", lcc98.units = "%";
        mcc99.layer = 99, mcc99.name = "Medium Cloud Cover", mcc99.units = ;
        hcc100.layer = 100, hcc100.name = "High Cloud Cover", hcc100.units = ;
        tcc101.layer = 101, tcc101.name = "Total Cloud Cover", tcc101.units = ;
        pres102.layer = 102, pres102.name = "Pressure", pres102.units = ;
        gcc103.layer = 103, gcc103.name = "Geopotential Height", gcc103.units = ;
        gh104.layer = 104, gh104.name = "Pressure", gh104.units = ;
        pres105.layer = 105, pres105.name = "Pressure", pres105.units = ;
        gh106.layer = 106, gh106.name = "Geopotential Height", gh106.units = ;
        ulwrf107.layer = 107, ulwrf107.name = "Upward long-wave radiation flux", ulwrf107.units = ;
        dswrf108.layer = 108, dswrf108.name = "Downward short-wave radiation flux", dswrf108.paramId = 260087, dswrf108.shortName = "dswrf", dswrf108.units = "W m**-2";
        dlwrf109.layer = 109, dlwrf109.name = "Downward long-wave radiation flux", dlwrf109.paramId = 260097, dlwrf109.shortName = "dlwrf", dlwrf109.units = "W m**-2";
        uswrf110.layer = 110, uswrf110.name = "Upward short-wave radiation flux", uswrf110.paramId = 260088, uswrf110.shortName = "uswrf", uswrf110.units = "W m**-2";
        ulwrf111.layer = 111, ulwrf111.name = "Upward long-wave radiation flux", ulwrf111.paramId = 260098, ulwrf111.shortName = "ulwrf", ulwrf111.units = "W m**-2";
        vbdsf112.layer = 112, vbdsf112.name = "Visile Beam Downward Solar Flux", vbdsf112.paramId = 260346, vbdsf112.shortName = "vbdsf", vbdsf112.units = "W m**-2";
        vddsf113.layer = 113, vddsf113.name = "Visible Diffuse Downward Solar Flux", vddsf113.paramId = 260347, vddsf113.shortName = "vddsf", vddsf113.units = "W m**-2";
        uswrf114.layer = 114, uswrf114.name = "Upward short-wave radiation flux", uswrf114.units = ;
        hlcy115.layer = 115, hlcy115.name = "Storm relative helicity", hlcy115.paramId = 260125, hlcy115.shortName = "hlcy", hlcy115.units = "m**2 s**-2";
        hlcy116.layer = 116, hlcy116.name = "Storm relative helicity", hlcy116.units = ;
        ustm117.layer = 117, ustm117.name = "U-component storm motion", ustm117.units = ;
        vstm118.layer = 118, vstm118.name = "V-component storm motion", vstm118.units = ;
        vucsh119.layer = 119, vucsh119.name = "Vertical u-component shear", vucsh119.units = ;
        vvcsh120.layer = 120, vvcsh120.name = "Vertical v-component shear", vvcsh120.units = ;
        vucsh121.layer = 121, vucsh121.name = "Vertical u-component shear", vucsh121.paramId = 3045, vucsh121.shortName = "vucsh121h", vucsh121.units = "s**-1";
        vvcsh122.layer = 122, vvcsh122.name = "Vertical v-component shear", vvcsh122.units = ;
        gh123.layer = 123, gh123.name = "Geopotential Height", gh123.units = ;
        r124.layer = 124, r124.name = "Relative Humidity", r124.units = ;
        pres125.layer = 125, pres125.name = "Pressure", pres125.paramId = 54, pres125.shortName = "pres", pres125.units = "Pa";
        gh126.layer = 126, gh126.name = "Geopotential Height", gh126.units = ;
        r127.layer = 127, r127.name = "Relative Humidity", r127.units = ;
        pres128.layer = 128, pres128.name = "Pressure", pres128.paramId = 54, pres128.shortName = "pres", pres128.units = "Pa";
        gh129.layer = 129, gh129.name = "Geopotential Height", gh129.units = ;
        gh130.layer = 130, gh130.name = "Geopotential Height", gh130.units = ;
        fourlftx131.layer = 131, fourlftx131.name = "Best (4-layer) lifted index", fourlftx131.units = ;
        cape132.layer = 132, cape132.name = "Convective available potential energy", cape132.units = ;
        cin133.layer = 133, cin133.name = "Convective inhibition", cin133.units = ;
        hpbl134.layer = 134, hpbl134.name = "Planetary boundary layer height", hpbl134.units = ;
        gh135.layer = 135, gh135.name = "Geopotential height", gh135.units = ;
        cape136.layer = 136, cape136.name = "Convective available potential energy", cape136.units = ;
        cin137.layer = 137, cin137.name = "Convective inhibition", cin137.units = ;
        cape138.layer = 138, cape138.name = "Convective available potential energy", cape138.units = ;
        cin139.layer = 139, cin139.name = "Convective inhibition", cin139.units = ;
        gh140.layer = 140, gh140.name = "Geopotential Height", gh140.units = ;
        plpl141.layer = 141, plpl141.name = "Pressure of level from which parcel was lifted", plpl141.units = ;
        ATT142.layer = 142, ATT142.name = "unkn", ATT142.units = ;
        lsm143.layer = 143, lsm143.name = "Land-sea mask", lsm143.units = ;
        ci144.layer = 144, ci144.name = "Sea ice area fraction", ci144.units = ;
        sbt123145.layer = 145, sbt123145.name = "Simulated Brightness Temperature for GOES 12, Channel 3", sbt123145.units = ;
        sbt124146.layer = 146, sbt124146.name = "Simulated Brightness Temperature for GOES 12, Channel 4", sbt124146.units = ;
        sbt113147.layer = 147, sbt113147.name = "Simulated brightness Temperature for GOES 11, Channel 3", sbt113147.units = ;
        sbt114148.layer = 148, sbt114148.name = "Simulated Brightness Temperature for GOES 11, Channel 4", sbt114148.units = ;





        Parameter objparamArr[numParams] = {refc1, ATT2, veril3, vis4, refd5, refd6, refd7, gust8, u9, v10, u11, v12, gh13,
                  t14, dpt15, u16, v17, gh18, t19, dpt20, u21, v22, gh23, t24, dpt25, u26, 
                  v27, t28, dpt29, u30, v31, t32, dpt33, u34, v35, ATT37, wz38, msla39, gh40,
                  ATT41, refd42, ATT43, ATT44, ATT45, ATT46, ATT47, ATT48, vo49, vo50, hail51,
                  hail52, ATT53, ltng54, u55, v56, sp57, orog58, t59, asnow60, mstav61,cnwat62,
                  sdwe63, snowc64, sde65, twot66, pt67, twosh68, twod69, twor70, tenu71, tenv72,
                  tensi73, ATT74, ATT75, cpofp76, prate77, tp78, sdwe79, ATT80, frzr81, ssrun82,
                  bgrun83, csnow84, cicep85, cfrzr86, crain87, sr88, fricv89, shtfl90, lhtfl91,
                  gflux92, vgtyp93, lftx94, cape95, cin96, pwat97, lcc98, mcc99, hcc100, tcc101,
                  pres102, gh103, gh104, pres105, gh106, ulwrf107, dswrf108, dlwrf109, uswrf110,
                  ulwrf111, vbdsf112, vddsf113, uswrf114, hlcy115, hlcy116, ustm117, vstm118,
                  vucsh119, vvcsh120, vucsh121, vvcsh122, gh123, r124, pres125, gh126, r127,
                  pres128, gh129, gh130, fourlftx131, cape132, cin133, hpbl134, gh135, cape136,
                  cin137, cape138, cin139, gh140, plpl141, ATT142, lsm143, ci144, sbt123145,
                  sbt124146, sbt113147, sbt114148};

        int layer =0;
        for (int i = 0; i<numParams; i++){
            layer = objparamArr[i].layer;
            blnParamArr[layer] = true;
        }        

        stationArr = new Station[numStations];

        Station bmtn, ccla, farm, huey, lxgn;
        bmtn.name = "BMTN";bmtn.lat = 36.91973;bmtn.lon = -82.90619;
        *(stationArr+0) = bmtn;
        ccla.name = "CCLA";ccla.lat = 37.67934; ccla.lon = -85.97877;
        *(stationArr+1) = ccla;
        farm.name = "FARM"; farm.lat = 36.93; farm.lon = -86.47;
        *(stationArr+2) = farm;
        huey.name = "HUEY"; huey.lat = 38.96701; huey.lon = -84.72165;
        *(stationArr+3) = huey;
        lxgn.name = "LXGN"; lxgn.lat = 37.97496; lxgn.lon = -84.53354;
        *(stationArr+4) = lxgn;

        // initialize the pointer array for each station to be of the length of the number of params
        for (int i=0; i<numStations;i++){
            stationArr[i].values = new double[numParams];
        }


        // build the hour array
        // make sure correct values have been passed to the hour array 
        try{
            intHourRange = arrHourRange.at(1) - arrHourRange.at(0)+1;
            if(intHourRange < 1) throw(intHourRange);
        }catch(exception e){
            fprintf(stderr, "Error, problems with hour range.");
            exit(0);
        }

        hours = (string*)malloc(intHourRange*sizeof(string));
        if(!hours){
            fprintf(stderr, "Error, unable to allocate hours");
            exit(0);
        }
        
        int endHour = arrHourRange.at(1);
        int beginHour = arrHourRange.at(0);
        int index=0;
        for(int hour = beginHour; hour<=endHour;hour++){
            if(hour < 10){ // put a 0 in front then insert in the hours arr
                string strHour = "0"+to_string(hour);
                hours[index++] = strHour;
            }
            else{ // just convert to a string and insert into the hours arr
                string strHour = to_string(hour);
                hours[index++] = strHour;
            }
        }
    }
}

void convertLatLons(){
    for(int i=0; i<numStations;i++){
        Station *station = &stationArr[i];
        station->lon = (station->lon +360); // circle, going clockwise vs counterclockwise
    }
}

bool checkDateRange(vector<int> beginDate, vector<int> endDate){
	// check that the array is of the size expected // WORKS
	if((beginDate.size() != 3) || (endDate.size() != 3)) return false;

	// check that endDay is further than endDay
	if(beginDate.at(0)>endDate.at(0)) return false;
	else if(beginDate.at(0) == endDate.at(0)){
		if(beginDate.at(1) > endDate.at(1)) return false;
		else if(beginDate.at(1) == endDate.at(1)){
			if(beginDate.at(2) >= endDate.at(2)) return false;
		}
		
	}

	// check that they have actually passed a valid date

	return true;
}

// based on: https://www.studymite.com/cpp/examples/program-to-print-the-next-days-date-month-year-cpp/
vector<int> getNextDay(vector<int> beginDate){
	
	int day, month, year;
	day = beginDate.at(2);
	month = beginDate.at(1);
	year = beginDate.at(0);

	if (day > 0 && day < 28) day+=1; //checking for day from 0-27, can just increment
	else if(day == 28){
		if(month == 2){ // if it is february, need special care
			if ((year % 400 == 0) || (year % 100 != 0 || year % 4 == 0)){ // is a leap year
				day == 29;
			}
			else{
				day = 1;
				month = 3;
			}
		}
		else // its not feb
		{
			day += 1; 
		}
	}
	else if(day == 29) // last day check for feb on a leap year
	{
		if(month == 2){
			day = 1;
			month = 3;
		}
	}
	else if(day == 30) // last day check for april, june, sept, nov
	{
		if(month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12)
		{
			day += 1;
		}
		else
		{
			day = 1;
			month +=1;
		}
	}
	else if(day == 31) // last day of the month
	{
		day = 1;
		if(month == 12) // last day of the year
		{
			year += 1;
			month = 1;
		}
		else month+=1;
	}
	vector<int> nextDay = {year, month, day};
	return nextDay;
}


vector<string> formatDay(vector<int> date){
	string strYear, strMonth, strDay;
	int intYear = date.at(0);
	int intMonth = date.at(1);
	int intDay = date.at(2);

	if(intMonth < 10){
		strMonth = "0" + to_string(intMonth);
	}
	else strMonth = to_string(intMonth);
	if(intDay < 10) strDay = "0" + to_string(intDay);
	else strDay = to_string(intDay);
	strYear = to_string(intYear);

	vector<string> formattedDate = {strYear , strMonth, strDay , (strYear+strMonth+strDay)};
	return formattedDate;
}


bool dirExists(string filePath){
 	struct stat info;

	if(stat(filePath.c_str(), &info) != 0){
		return false;
	}
	else if(info.st_mode & S_IFDIR){
		return true;
	}
	else return false;
}


void readData(FILE* f, string fileName, string filePath2){

    printf("\nOpening File: %s", filePath2.c_str());
    // cout << "Opening file: " << filePath2 << endl;
    //try to open the file
    try{
        f = fopen(filePath2.c_str(), "rb");
        if(!f) throw(filePath2);
    }
    catch(string file){
        printf("Error: could not open filename %s in directory %s", fileName.c_str(), file.c_str());
        return;
    }
    unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
                                              CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
    // codes_grib_multi_support_on(NULL);

    codes_handle * h = NULL; // use to unpack each layer of the file

    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;

    char value[MAX_VAL_LEN];
    size_t vlen = MAX_VAL_LEN;

    bool flag = true; 

    long numberOfPoints=0;

    double *lats, *lons, *values; // lats, lons, and values returned from extracted grib file

    while((h=codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err))!=NULL) // loop through every layer of the file
    {
        assert(h);
        msg_count++; // will be in layer 1 on the first run
        if (blnParamArr[msg_count] == true){

            clock_gettime(CLOCK_MONOTONIC, &startExtract);
            // extract the data
            CODES_CHECK(codes_get_long(h, "numberOfPoints", &numberOfPoints), 0);
            CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);
            lats = (double*)malloc(numberOfPoints * sizeof(double));
            if(!lats){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                exit(0);
            }
            lons = (double*)malloc(numberOfPoints * sizeof(double));
            if (!lons){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                std::free(lats);
                exit(0);
            }
            values = (double*)malloc(numberOfPoints * sizeof(double));
            if(!values){
                fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
                std::free(lats);
                std::free(lons);
                exit(0);
            }
            CODES_CHECK(codes_grib_get_data(h, lats,lons,values), 0);
            // all the lats,lons, and values from this layer of the grib file are now stored, now match them
            // to their respective stations

            clock_gettime(CLOCK_MONOTONIC, &endExtract);
            double thisTime= (endExtract.tv_sec - startExtract.tv_sec) * 1000.0;
            thisTime+= (endExtract.tv_nsec - startExtract.tv_nsec) / 1000000.0;
            extractTime += thisTime;

            // if it is the first time, extract the index
            if (flag == true){
                clock_gettime(CLOCK_MONOTONIC, &startMatching);

                for(int i=0; i<numStations; i++){
                    Station *station = &stationArr[i];
                    int closestPoint = 0;
                    for(int j=0; j<numberOfPoints;j++){
                        double distance = pow(lats[j] - station->lat, 2) + pow(lons[j]-station->lon, 2);
                        double closestDistance = pow(lats[closestPoint] - station->lat, 2) + pow(lons[closestPoint] - station->lon, 2);
                        // if the distance calculated on this iteration is closer than the distance calculated on the closest point, then change the closest point to this iteration
                        if(distance < closestDistance) closestPoint = j;
                    }
                    station->closestPoint = closestPoint;
                }
                flag = false;
                clock_gettime(CLOCK_MONOTONIC, &endMatching);
                double timeeeeee = (endMatching.tv_sec - startMatching.tv_sec) * 1000.0;
                timeeeeee+=(endMatching.tv_nsec - startMatching.tv_nsec) / 1000000.0;
                matchTime+= timeeeeee;
            }

            // we've got the index of the closest lats and lons, now we just have to map them to each station's values arr
            // figuring out which index of each station's parameter array to put the values at 
            for (int i=0; i<numStations; i++){
                Station *station = &stationArr[i];
                int index = 0;
                for (int j =0; j<numParams;j++){
                    if(objparamArr[j].layer == msg_count){
                        index = j;
                        break;
                    }
                }
                *(station->values+index) = values[station->closestPoint]; 

            }

            std::free(lats);
            std::free(lons);
            std::free(values);
            
        }
        codes_handle_delete(h);

    }
    fclose(f);

}


void mapData(string date, string hour){
    string hourlyDate = date + hour;
    for (int i = 0; i<numStations; i++){
        Station *station = &stationArr[i];
        // we can't directly insert the pointer to the data into the dataMap, so we need to insert a copy of the array
        std::vector<double> valuesCpy(numParams);
        for(int j=0; j<numParams; j++){
            valuesCpy[j] = *(station->values+j);
        }
        station->dataMap.insert({hourlyDate, valuesCpy});
    }

}

void writeData(vector<string> strCurrentDay, string hour){
    // if the path does not exist, make the path
    if(!dirExists(writePath)){
        if(mkdir(writePath.c_str(), 0777) == -1){
            cerr << "Error: " << strerror(errno) << endl;
        }
    }

    // make the directory structure: years/days/stationData
    //                              ex. 2019/20190101/BMTN.20190101.csv

    // for each year, make a folder
    string yearWritePath = writePath+strCurrentDay.at(0)+"/";
    if(!dirExists(yearWritePath)){
        if(mkdir(yearWritePath.c_str(),0777)==-1){
            cerr << "Error: " << strerror(errno) << endl;
        }
    }
    // for each day, make a folder
    string dayWritePath = yearWritePath + strCurrentDay.at(3) + "/";
    if(!dirExists(dayWritePath)){
        if(mkdir(dayWritePath.c_str(), 0777)==-1){
            cerr << "Error: " << strerror(errno) << endl;
        }
    }
    // for each station, make a file
    for(int i=0; i<numStations; i++){
        Station *station = &stationArr[i];
        string fullFilePath = dayWritePath + station->name + "." + strCurrentDay.at(3) + ".csv";
        // if the file does not already exist, make it
        if(!std::filesystem::exists(fullFilePath)){

            // give the new file the appropriate headers
            ofstream outfile;
            outfile.open(fullFilePath, std::ios_base::app);
            outfile << "year, month, day, hour, ";
            
            // append the name of each parameter to the headings of the files
            for(int j=0;j<numParams;j++){
                outfile << " " << objparamArr[j].name << " (" << objparamArr[j].units << ")" <<",";
            }
            outfile << "\n";
            outfile.close();
        }

        // now append the data gathereed for the hour into the file
        ofstream csvFile;
        csvFile.open(fullFilePath, std::ios_base::app);
        // append the year,                         month,                      day,                         hour, 
        csvFile << strCurrentDay.at(0) << "," << strCurrentDay.at(1) << "," << strCurrentDay.at(2) << "," << hour << ",";

        // iterate through the vector at the yearMonthDayHour and append to the file
        string key = strCurrentDay.at(3) + hour;
        auto itr = station->dataMap.find(key.c_str());
        for(int j=0; j<itr->second.size();j++){
            csvFile << itr->second.at(j) << ",";
        }
        
        csvFile << "\n";
        csvFile.close() ;   
    }


}
