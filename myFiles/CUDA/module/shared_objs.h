#ifndef SHARED_OBJS_H
#define SHARED_OBJS_H

#include <stdio.h>
#include <semaphore.h>

typedef struct Station{
    const char* grid_idx;
    const char* state_fips;
    const char* stateAbbrev;
    const char* county;
    const char* fipsCode;
    float latll; // each grid index will include the lat and lons of 
    float lonll; // the lower left and upper right corners
    float latur;
    float lonur;
    double **values; // holds the values of the parameters. Index of this array will 
                    // correspond to index if the Parameter array. This will be a single hour's data
    int closestPoint;
} station_t;

typedef struct threadArgs{
    FILE* f;
    const char* pathName;
    int threadIndex;
    const char* hour;
    const char* strCurrentDay;
    bool first_hour_flag;
    bool last_hour_flag;
    sem_t *values_protection;
    sem_t *barrier;

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