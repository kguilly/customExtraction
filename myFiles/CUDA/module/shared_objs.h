#ifndef SHARED_OBJS_H
#define SHARED_OBJS_H

#include <stdio.h>

typedef struct Station{
    char* grid_idx = "00";
    char* state_fips;
    char* stateAbbrev;
    char* county;
    char* fipsCode;
    float latll; // each grid index will include the lat and lons of 
    float lonll; // the lower left and upper right corners
    float latur;
    float lonur;
    double **values; // holds the values of the parameters. Index of this array will 
                    // correspond to index if the Parameter array. This will be a single hour's data
    int* closestPoint;
} station_t;

typedef struct threadArgs{
    FILE* f;
    char* fileName;
    char* pathName;
    int threadIndex;
    char* hour;
    char* strCurrentDay;
} threadArgs_t;

typedef struct DeviceInfo {
    char* name;
    int clockRate;
    int maxBlocksperMulti;
    int maxThreadsperBlock;
    size_t free_mem;
    size_t total_mem;
} deviceInfo_t;

#endif