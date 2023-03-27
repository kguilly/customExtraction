#ifndef DECOMPRESS_CUDA_H
#define DECOMPRESS_CUDA_H

#include <cuda_runtime.h>
#include "shared_objs.h"
#include "decompress_funcs.h"

deviceInfo_t device_information (int);

__global__ void find_nearest_points (station_t*, double*, double*, int, int);

void index_extraction (station_t*, double*, double*, int, int);

__global__ void get_station_values (int, int, int, int, double*);

void * read_grib_data(void*);

#endif /* DECOMPRESS_CUDA_H */

