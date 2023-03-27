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
void defaultStations(); 
void readCountycsv(); 
void matchState();
void getStateAbbreviations();
// similar to default station, but for the parameter arrays
void defaultParams(bool);

void buildHours();// builds the hour arrays given the hour range specified

void semaphoreInit(); // initialize all semaphores used

/* function to convert the lats and lons from the passed representation to the 
    way they are represented in the grib file. ONLY WORKS for coordinates in USA*/
void convertLatLons();

/*Function to create the paths and files for the maps to be written to*/
void createPath();

/* function to format the date as a vector of strings
   returns vector<string> date = {yyyy, mm, dd, yyyymmdd}*/
std::vector<std::string> formatDay(std::vector<int>);

/* Function that splits a given string on a specified delimiter */
std::vector<std::string> splitonDelim(std::string, char);

/*Find the length of a given string*/
int len(std::string);

// function to check the begin day vs the end day to see if a valid range has been passed
bool checkDateRange(std::vector<int>, std::vector<int>);

// function to get the next day after the day passed
std::vector<int> getNextDay(std::vector<int>);

/*function to optimize the work taken to find the indexes*/
void get_nearest_indexes(std::vector<std::string>, double*, double*, long*);

/* function to check if a given directory exists*/
bool dirExists(std::string);

/*function to write the data after each day is extracted*/
void write_daily_data(std::vector<std::string>);

void garbageCollection();

int main(int, char**);

#endif