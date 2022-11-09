/*
This file will work off of the example provided in the eccodes library
to extract grib files, and follow the methodolgy used in the ExtractWRFData.py
to index files and locations for Mesonet stations correctly. 

This first iteration will mainly focus on the data extraction, not file output

**Note: This iteration only extracts the first parameter, not all

Compile: g++ -Wall extractWRFData1.cpp -leccodes

COMMAND LINE ARGS:
    ./outfile.out -begin (date) -end (date)     :   Dates are formatted as YYYYmmdd, 
                                                    ex: ./a.out -begin 20190101 -end 20190102
*/

#include <stdio.h>
#include <stdlib.h>
#include "eccodes.h"
#include <time.h>
#include <iostream>
#include <cstring>

void validateInput(char *startDay, char *endDay);
char* getFilePath();
char* readData();

//init vars
char* startDay = "20190101"; // format as "YYYYmmdd" , ex. "20190203" as february 3, 2019
char* endDay = "20190102"; // NOT inclusive (doesn't include the last day)
// point at the "data folder that contains the years 
const char *filePath = "/home/kalebg/Desktop/School/Y4S1/REU/customExtraction/UtilityTools/extractTools/data/"; 
// rest of file path: 2019/20190101/hrrr.20190101.00.00.grib2
int err = 0;
size_t i = 0;
FILE* gribFile = NULL;
codes_handle* h = NULL; // structure giving access to parsed values by keys
long numberOfPoints = 0;
const double missing = 1.0e36; // the value assigned to a missing value 
double *lats, *lons, *values; // store the values in arrays rather than dictionaries
clock_t start, end;
double cpuTimeUsed;

using namespace std;
int main(int argc, char**argv){
    
    // validateInput(startDay, endDay); keep getting seg faults, come back to it later
    // validate arguments
    if(argc > 5){
        cout << "ERROR: too many arguments." << endl;
    }
    for (int i =1; i<argc ; i++){
        if(strcmp(argv[i], "-begin")==0){
            if(argv[i+1]!=NULL){
                startDay = argv[i+1];
            }else{
                cout << "ERROR: need another argument." << endl;
                return 1;
            }
        }
        else if(strcmp(argv[i], "-end")==0){
            if(argv[i+1]!=NULL){
                endDay = argv[i+1];
            }else{
                cout << "ERROR: Need to pass another argument." << endl;
                return 1;
            }
        }

    }
    int intStartDay; 
    int intEndDay;

    sscanf(startDay, "%d", &intStartDay);
    sscanf(endDay, "%d", &intEndDay);
    int dateRange = intEndDay - intStartDay; // not correct, will fix later
    if(dateRange < 0){
        cout << "ERROR: begin date - end date must be greater than 0" << endl;
        return 1;
    }
    

    for (int i=0; i<24 ; i++){
        char year[4];
        for(int j=0; j<4; j++){
            year[j] = startDay[j]; 
        }
        int hour = i;
        // can't figure out how to convert this pointer array to concat with the mf
        // char* fileName = strcat(filePath1)
        // seg faults
        // string filee = strcat(year, strcat("/", strcat(startDay, 
        //                 strcat("/hrrr.", strcat(startDay, strcat(".", strcat((char*)hour,".00.grib2")))))));
        //string filee2 = strcat((char*)filee, startDay);
        

        // construct char* array for the entire file path. 
        //"/home/kalebg/Desktop/School/Y4S1/REU/customExtraction/UtilityTools/extractTools/data/2019/20190101/hrrr.20190101.00.00.grib2
        /*
        year is 4 spaces, '/' is one space, 
        // int filePathLength = sizeof(filePath)+sizeof(year)+1+sizeof(startDay)+6+sizeof(startDay)+1+sizeof(hour)+9;
        */
        int filePathLength = 0;
        char filepathiterator = filePath[0];
        while (filepathiterator != '\0'){
            filePathLength++;
            filepathiterator = filePath[filePathLength];
        }
        filePathLength+=39;

    }

    return 0;

}

/*
// void validateInput(char *startDay, char * endDay){
//     // catch error if dateRange is less than 1
//     cout << "Enter the start date (mm/dd/YYYY): ";
//     cin >> startDay;

//     while(1){
//         if(cin.fail()){
//             cin.clear();
//             cin.ignore(numeric_limits<streamsize>::max(), '\n');
//             cout << "Invalid, try again: ";
//             cin >> startDay;
//         }
//         else if(strlen(startDay) != 10 || startDay[2] != '/' || startDay[5] != '/'){ // or if the day is not formatted correctly
//             cin.clear();
//             cin.ignore(numeric_limits<streamsize>::max(), '\n');
//             cout << "Invalid format, try again: ";
//             cin >> startDay;
//         }
//         else if(!cin.fail()){
//             break;
//         }
//     }

//     cout << "Enter the end date (mm/dd/YYYY): ";
//     cin >> endDay;

//     while(1){
//         if(cin.fail()){
//             cin.clear();
//             cin.ignore(numeric_limits<streamsize>::max(), '\n');
//             cout << "Invalid, try again: ";
//             cin >> endDay;
//         }
//         else if(strlen(endDay) != 10 || endDay[2] != '/' || endDay[5] != '/'){
//             cin.clear();
//             cin.ignore(numeric_limits<streamsize>::max(), '\n');
//             cout << "Invalid format, try again: ";
//             cin >> startDay;
//         }
//         else if(!cin.fail()){
//             break;
//         }
//     }

// }
*/