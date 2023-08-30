#ifndef DECOMPRESS_GRIB_TEST_H
#define DECOMPRESS_GRIB_TEST_H

#include <cuda_runtime.h>
#include "shared_test_objs.h"

station_t* extract_indexes(station_t*, double*, double*, int, int);
__global__ void cuda_find_nearest(station_t*, double*, double*, int, int);
station_t* cuda_orchestrate_decompress_grib(station_t*, const char*, char***, int, int, int);
__global__ void cuda_decompress_grib(station_t*, const char*, const char***, int,  int, int, int*);
__device__ void cuda_match_station_to_vals(station_t*, double*, int, int, int);
__device__ long charToLong (const char*);
__device__ int charToInt (const char*);

#endif

