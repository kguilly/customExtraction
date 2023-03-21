#ifndef DECOMPRESS_FUNCS_H
#define DECOMPRESS_FUNCS_H

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cstring>

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
bool dirExists(string);

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

/*function to write the data after each day is extracted*/
void writeHourlyData(bool, vector<string>);

void garbageCollection();

int main(int*, char*[]);