/*
This file will work off of the example provided in the eccodes library
to fit our extraction of grib files :: /eccodes/examples/grib_get_data.c
- kaleb Guillot
*/

#include <stdio.h>
#include <stdlib.h>
#include "eccodes.h"
#include <time.h>

/*
All references to eccodes functions are undefined
*/

int main(){
    // init vars
    int err = 0;
    size_t i = 0;
    FILE* gribFile = NULL;
    const char* filePath = "/home/kalebg/Desktop/School/Y4S1/REU/customExtraction/UtilityTools/extractTools/data/2019/20190101/hrrr.20190101.00.00.grib2"; 
    codes_handle* h = NULL; // structure giving access to parsed values by keys
    long numberOfPoints = 0;
    const double missing = 1.0e36; // the value assigned to a missing value 
    double *lats, *lons, *values; // store the values in arrays rather than dictionaries
    clock_t start, end;
    double cpuTimeUsed;

    start = clock();
    gribFile = fopen(filePath, "rb");
    if(!gribFile){
        fprintf(stderr, "Error: unable to open input file %s\n", filePath);
        return 1;
    }

    // create new handle from a message in a file // 
    h = codes_handle_new_from_file(0, gribFile, PRODUCT_GRIB, &err);
    if (h==NULL){
        fprintf(stderr, "Error: unable to create handle from file %s\n", filePath);
        return 1;
    }

    CODES_CHECK(codes_get_long(h, "numberOfPoints", &numberOfPoints), 0);
    CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);
    lats = (double*)malloc(numberOfPoints * sizeof(double));
    if(!lats){
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
        return 1;
    }
    lons = (double*)malloc(numberOfPoints * sizeof(double));
    if (!lons){
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
        free(lats);
        return 1;
    }
    values = (double*)malloc(numberOfPoints * sizeof(double));
    if(!values){
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
        free(lats);
        free(lons);
        return 1;
    }

    CODES_CHECK(codes_grib_get_data(h, lats, lons, values), 0);

    for(i=0; i<numberOfPoints; ++i){
        if(values[i] != missing){
            // TESTING DECOMPRESSING VALUES AT CERTAIN COORDINATES
            printf("%f %f %f\n", lats[i], lons[i], values[i]);
        }
    }
    // Track time for file to decompress a single file 
    end = clock();
    cpuTimeUsed = ((double)(end-start)) / CLOCKS_PER_SEC; // in seconds
    printf("Time to decompress a single GRIB file: %.6f", cpuTimeUsed);

    free(lats);
    free(lons);
    free(values);
    codes_handle_delete(h);

    fclose(gribFile);
    return 0;


}
