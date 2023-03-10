#include <iostream>
#include <string>
#include <stdio.h>
#include "grib_api.h" // grib header file

/*
Compile and run
- using gdb
g++ -g -Wall importGribTest.cpp ; gdb a.out
*/
using namespace std;
int main(){

    cout << "Successfully imported the grib header file" << endl;

    // now try to call a function from grib
    char file_path[] = "UtilityTools/extractTools/data/2019/20190101/hrrr.20190101.00.00.grib2"; 
    
    // need to make file obj from file path
    FILE *file = fopen((file_path), "r");
    
    grib_handle * gribFile = grib_handle_new_from_file(NULL, file, NULL);
    // grib_handle * gribFile = grib_handle_new_from_file(NULL, file, NULL); 
    
    char out_file_path[] = "outfiles/outfile.csv";
    FILE * outfile =  fopen(out_file_path, "w");
    // copy the grib dump content over 
    grib_dump_content(gribFile, outfile, NULL, 0, NULL);


    return 0;
}

/*Error Log:

Undefined references to my grib functions
Fix: need to import eccodes.h instead of grib_api

*/
