#ifndef DECOMPRESS_GRIB_TEST_H
#define DECOMPRESS_GRIB_TEST_H

#include <cuda_runtime.h>
#include "shared_test_objs.h"

station_t* orchestrate_grib_decompression(station_t*, FILE*, int, int, 
                                          long, bool*, size_t);
__global__ void cuda_decompress_grib(station_t*, FILE*, double*, double*,
                                     double*, codes_handle*, int, 
                                     long, bool*, int);
__device__ void match_station_values(station_t*, double*, int, int);


station_t* extract_indexes(station_t *, double*, double*, int, int);
__global__ void cuda_find_nearest(station_t*, double*, double*, int, int);



#endif