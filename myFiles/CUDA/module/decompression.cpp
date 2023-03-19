#include <stdio.h>
#include <stdlib.h>
#include "eccodes.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector> 
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include <time.h>
#include <cassert>
#include <cmath>
#include "semaphore.h"

// Personal Headers
#include "shared_objs.h"
#define MAX_VAL_LEN 1024

using namespace std;
/* Timing variables */
struct timespec startTotal;
struct timespec endTotal;
double totalTime;

vector<int> beginDay = {2023, 1, 31}; // arrays for the begin days and end days. END DAY IS NOT INCLUSIVE.
                                     // when passing a single day, pass the day after beginDay for endDay
                                     // FORMAT: {yyyy, mm, dd}
vector<int> endDay = {2023, 2, 3};   // NOT INCLUSIVEe

vector<int> arrHourRange = {0,23}; // array for the range of hours one would like to extract from
                                   // FORMAT: {hh, hh} where the first hour is the lower hour, second is the higher
                                   // accepts hours from 0 to 23 (IS INCLUSIVE)

int intHourRange; 

string filePath = "/media/kaleb/extraSpace/wrf/";  // path to "data" folder. File expects structure to be:
                                        // .../data/<year>/<yyyyMMdd>/hrrr.<yyyyMMdd>.<hh>.00.grib2
                                        // for every hour of every day included. be sure to include '/' at end

string writePath = "/home/kaleb/Desktop/WRFextract_2-3/"; // path to write the extracted data to,
                                                    // point at a WRFData folder
string repositoryPath = "/home/kaleb/Documents/GitHub/customExtraction/";//PATH OF THE CURRENT REPOSITORY
                                                                          // important when passing args                                                    

Station *stationArr; 
                        // this will be used to quickly index whether a parameter needs to be 
                        // extracted or not. Putting 149 spaces for 148 parameters because the
                        // layers of the parameters start at 1 rather than 0
vector<string> vctrHeader;
int numStations, numParams;
bool* blnParamArr;

// 24 hours in a day. Use this to append to the file name and get each hour for each file
string *hours;

/*Initialization of semaphores to be used for protection*/
sem_t hProtection; // protect when creating the handle object for each layer
sem_t *mapProtection; // protect when writing to the maps
sem_t pathCreationSem; // protect when writeData makes new file paths
sem_t *valuesProtection; // protect when writing values to the values array
sem_t writeProtection; // only one thread should be able to write to a file at a time
sem_t headerProtection; // only one thread can read the header flag at a time

// function to handle arguments passed. Will either build the Station array and/or paramter array
// based off of the arguments passed or will build them with default values. Will also construct
// the hour array based on passed beginning and end hours
void handleInput(int, char**);

// Function to build the default station array (all counties in continential US)
// through reading from the files in the countyInfo file
void defaultStations(); void readCountycsv(); void matchState();
void getStateAbbreviations();
// similar to default station, but for the parameter arrays
void defaultParams(bool, int, char**);

void buildHours();// builds the hour arrays given the hour range specified

void semaphoreInit(); // initialize all semaphores used

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

/* function to write the parameter information for a given day out to csv*/
void paramInfotoCsv(vector<string>);

/*function to optimize the work taken to find the indexes*/
void get_nearest_indexes(vector<string>);

/* function to read the data from a passed grib file */
void *readData(void*);

/*Function to find the standard deviation of the values for each week,
takes an array of the values for the week and their average, outputs a stdDev*/
static double standardDev(vector<double>, double&);

/*Function to create the paths and files for the maps to be written to*/
void createPath();

/*functions to help create paths by finding length of string and splitting strnig on delimeter*/
int len(string); vector<string> splitonDelim(string, char);

/*function to write the data after each day is extracted*/
void writeHourlyData(bool, vector<string>);

void garbageCollection();

int main(int argc, char*argv[]){
    clock_gettime(CLOCK_MONOTONIC, &startTotal);

    // function calls to initialize everything
    handleInput(argc , argv);
    semaphoreInit();
    convertLatLons();

    // validate the dates passed
    bool boolDateRange = checkDateRange(beginDay, endDay);
    if (boolDateRange == false){
        fprintf(stderr, "Invalid Date Range\n");
        exit(1);
    }
    
    // before writing to anything, make sure all necessary directories exist
    createPath();
    vector<int> intCurrentDay = beginDay;
    string prevMonth = "";
    
    while(checkDateRange(intCurrentDay, endDay)){
        
        vector<string> strCurrentDay = formatDay(intCurrentDay);
        string currMonth = strCurrentDay.at(1);
        if (currMonth != prevMonth) {
            prevMonth = currMonth;
            cout << "Getting Station Indexes for Date " << strCurrentDay.at(3);
            get_nearest_indexes();
        }
    }
}

void get_nearest_indexes(vector<string> strCurrentDay){
    /*
    There are several goals of this function
        - Open a grib file from the day passed
        - Grab the header to write out to the csv files
        - Find the nearest indexes for the corresponding stations
    */

    // concatenate the file path
    string filePath1 = filePath + strCurrentDay.at(0) + "/" + strCurrentDay.at(3) + "/";
    // check if the file path exists
    if(dirExists(filePath1) == false){
        fprintf(stderr, "Error: could not find directory %s", filePath1.c_str());
        exit(1);
    }

    // in the directory, find a suitable grib file to open and read the index
    string year = strCurrentDay.at(0);
    string month = strCurrentDay.at(1);
    string day = strCurrentDay.at(2);
    string hour, grib_file_path;
    FILE* f;
    for (int i=0; i<intHourRange; i++){
        hour = hours[i];
        grib_file_path = filePath1 + year + "/" + year + month + day + "/hrrr." + year + \
                         month + day + "." + hour + ".00.grib2"; 

        // now try to open the file. If it cannot, then go to the next iteration
        try{
            f = fopen(grib_file_path.c_str(), "rb");
            if (!f) throw(grib_file_path);
            else break;
        }
        catch (string file) {
            continue;
        }
    }

    // Once the file has been found and opened, it's time to read from it and
    // - get the header
    // - get the nearest indexes
    // - get the total parameter count ? maybe
    /////////////////
    // init 
    codes_grib_multi_support_on(NULL);
    codes_handle * h = NULL; // use to unpack each layer of the file
    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;
    size_t vlen = MAX_VAL_LEN;
    char value_1[MAX_VAL_LEN];
    bool flag = true; 
    long numberOfPoints=0;
    double *grib_lats, *grib_lons, *grib_values;
    string name_space = "parameter";
    unsigned long key_iterator_filter_flags = CODES_KEYS_ITERATOR_ALL_KEYS |
                                              CODES_KEYS_ITERATOR_SKIP_DUPLICATES;
    
    while((h = codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err)) != NULL){
        msg_count++;

        if (blnParamArr[msg_count] == false) continue;

        CODES_CHECK(codes_get_long(h, "numberOfPoints", &numberOfPoints), 0);
        CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);
        grib_lats = (double*)malloc(numberOfPoints * sizeof(double));
        if(!grib_lats){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
            exit(0);
        }
        grib_lons = (double*)malloc(numberOfPoints * sizeof(double));
        if (!grib_lons){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
            std::free(grib_lats);
            exit(0);
        }
        grib_values = (double*)malloc(numberOfPoints * sizeof(double));
        if(!grib_values){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
            std::free(grib_lats);
            std::free(grib_lons);
            exit(0);
        }

        CODES_CHECK(codes_grib_get_data(h, grib_lats, grib_lons, grib_values), 0);

        // add the information to the header
        codes_keys_iterator* kiter = codes_keys_iterator_new(h, key_iterator_filter_flags, name_space.c_str());
        if (!kiter) {
            fprintf(stderr, "Error: unable to create keys iterator while reading params\n");
            exit(1);
        }
        string strUnits, strName, strValue, strHeader, strnametosendout;
        while(codes_keys_iterator_next(kiter)){
            const char*name = codes_keys_iterator_get_name(kiter);
            vlen = MAX_VAL_LEN;
            memset(value_1, 0, vlen);
            CODES_CHECK(codes_get_string(h, name, value_1, &vlen), name);
            strName = name, strValue = value_1;
            if(strName.find("name")!=string::npos){
                strValue.erase(remove(strValue.begin(), strValue.end(), ','), strValue.end());
                //strHeader.append(to_string(msg_count)+"_"+strValue);
                strnametosendout = strValue;
            }
            else if(strName.find("units")!=string::npos){
                //strHeader.append("("+strValue+")");
                strUnits = "(" + strValue + ")";
            }
        }
        strHeader = strnametosendout + " " + strUnits;
        vctrHeader.push_back(strHeader);
        
        // if this flag is set, then grab the nearest indexes
        if (flag) {
            flag = false;
            // TODO: make a cuda kernel to parallelize this
        }
    }
}