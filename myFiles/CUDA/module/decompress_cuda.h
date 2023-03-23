#ifndef DECOMPRESS_CUDA_H
#define DECOMPRESS_CUDA_H

#include <cuda_runtime.h>
#include "shared_objs.h"

deviceInfo_t device_information(int);

__global__ void find_nearest_points (station_t*, double*, double*, int, int);

void index_extraction (station_t*, double*, double*, int, int);

#endif /* DECOMPRESS_CUDA_H */

