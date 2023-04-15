#ifndef DECOMPRESS_GRIB_TEST_H
#define DECOMPRESS_GRIB_TEST_H

#include <cuda_runtime.h>
#include "shared_test_objs.h"

station_t* extract_indexes(station_t *, double*, double*, int, int);
__global__ void cuda_find_nearest(station_t*, double*, double*, int, int);



#endif