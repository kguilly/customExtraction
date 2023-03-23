#include "decompress_cuda.h"
#include "iostream"
#include <stdlib.h>

/*
Funtion that gets device information:
 - number of available GPUs
 - max number of threads per block
*/

deviceInfo_t device_information (int device_num) {

    cudaSetDevice(device_num);
    cudaDeviceProp gpu;
    cudaGetDeviceProperties(&gpu, 0);
    deviceInfo_t dev;
    dev.name = gpu.name;
    dev.clockRate = gpu.clockRate;
    dev.maxThreadsperBlock = gpu.maxThreadsPerBlock;
    dev.maxBlocksperMulti = gpu.maxBlocksPerMultiProcessor;
    return dev;
}

/*
Function that performs the index extraction
*/
__global__ void find_nearest_points (station_t * stationArr, double * lats, double * lons, int numStations, int num_lons_lats) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numStations) {
        double min_distance = 999;
        int min_index = -1;
        
        station_t * curr_station = &stationArr[id];
        double st_lat = (curr_station->latll + curr_station->latur) / 2.0;
        double st_lon = (curr_station->lonll + curr_station->lonur) / 2.0;

        for (int i=0; i<num_lons_lats; i++) {
            double lat = lats[i];
            double lon = lons[i]; 
            double distance = sqrt(pow((st_lat - lat), 2) + pow((st_lon - lon), 2));

            if (distance < min_distance) {
                min_distance = distance;
                min_index = i;
            }
        }
        curr_station->closestPoint = min_index;
    }
    
}

/*
Function that orchestrates the index extraction
*/
void index_extraction (station_t * stationArr, double* lats, double* lons, int numStations, int numberOfPoints) { 
    
    // get useful device properties
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    deviceInfo_t gpu = device_information(0);
    cudaSetDevice(0);
    int max_threads = gpu.maxThreadsperBlock;
    int max_blocks = gpu.maxBlocksperMulti;

    int num_threads_to_use = max_threads / 2;
    int num_blocks_to_use = max_blocks / 2;

    // make device copies of host params
    station_t* d_stationArr;
    double* d_lats;
    double* d_lons;
    

    // allocate each of the arguments to the GPU
    if (cudaMalloc(&d_stationArr, sizeof(station_t) * numStations) != cudaSuccess) {
        std::cout << "stationArr could not be allocated to GPU" << std::endl;
        return;
    }
    if (cudaMalloc(&d_lats, sizeof(double) * numberOfPoints) != cudaSuccess) {
        std::cout << "Grib lats could not be allocated to the GPU" << std::endl;
        cudaFree(d_stationArr);
        return;
    }
    if (cudaMalloc(&d_lons, sizeof(double) * numberOfPoints) != cudaSuccess) {
        std::cout << "Grib lons could not be allocated to the GPU" << std::endl;
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        return;
    }

    // copy each of the arguments over to the gpu
    if (cudaMemcpy(d_stationArr, stationArr, sizeof(station_t) * numStations, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The stationArr could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    if (cudaMemcpy(d_lats, lats, sizeof(double) * numberOfPoints, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The grib lats could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    if (cudaMemcpy(d_lons, lons, sizeof(double) * numberOfPoints, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The grib lons could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    
    // call the kernel
    find_nearest_points <<< num_blocks_to_use, num_threads_to_use >>> (d_stationArr, d_lats, d_lons, numStations, numberOfPoints);
    cudaDeviceSynchronize();

    // copy the elements from the GPU back over to the host
    if (cudaMemcpy(stationArr, d_stationArr, sizeof(station_t) * numStations, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    std::cout << "cuda success" << std::endl;
    // release
    cudaFree(d_stationArr);
    cudaFree(d_lats);
    cudaFree(d_lons);

}
