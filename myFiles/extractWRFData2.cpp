/*
This file will work off the example provided in the eccodes library
:: "grib_print_data.c"
to extract data from grib files. 

This iteration will attempt to extract all parameters from a given file

Compile:
g++ -Wall extractWRFData.cpp -leccodes

RESULT: same as version 1, only extracts 1.9 million values rather than
1.9 mil * 150

*/

#include <stdio.h>
#include <stdlib.h>
#include "eccodes.h"
#include <time.h>
#include <iostream>
#include <cstring>  

const char* filePath = "/home/kalebg/Desktop/School/Y4S1/REU/customExtraction/UtilityTools/extractTools/data/2019/20190101/hrrr.20190101.00.00.grib2";
int err = 0;
size_t i = 0;
size_t values_len = 0;
FILE*gribFile = NULL;
codes_handle * h = NULL;
long numberOfPoints = 0, parameterNumber=0;
const double missing = 1.0e36; 
double *lats, *lons, *values;

static void usage(const char* prog){
    printf("usage: %s fileanme \n", prog);
    exit(1);
}

using namespace std;
int main(){
    gribFile = fopen(filePath, "rb");
    if(!gribFile){
        fprintf(stderr, "Error: unable to open input file %s\n", filePath);
        return 1;
    }

    /******* turn on supportrestaraunt ja for GRIB2 multi-field messages *********/
    codes_grib_multi_support_on(NULL);

    int paramCount = 0, paramIncludedCount = 0;
    while ((h = codes_handle_new_from_file(0, gribFile, PRODUCT_GRIB, &err))!= NULL) {
        CODES_CHECK(err, 0);
        // Get the parameter Number 
        CODES_CHECK(codes_get_long(h, "parameterNumber", &parameterNumber), 0);
        printf("\n\nParameterNumber=%ld\n", parameterNumber);
        paramCount++;


        paramIncludedCount++;
        // allocate the size of the values array // 
        CODES_CHECK(codes_get_size(h, "values", &values_len), 0); // get teh size of the values arr
        values = (double*)malloc(values_len*sizeof(double));

        /* get data values */
        CODES_CHECK(codes_get_double_array(h, "values", values, &values_len), 0);
        // print those values out 
        for(i = 0; i< values_len; i++){
            printf("%d %.10f\n", i, values[i]);
        }

        free(values);

        
        codes_handle_delete(h);
    }

    printf("\nThe number of parameters found in the file:: %d\n", paramCount);
    printf("The number of parameters actually included: %d\n", paramIncludedCount);
    fclose(gribFile);
    return 0;
}