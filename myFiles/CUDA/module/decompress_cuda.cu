#include "decompress_cuda.h"
#include "iostream"
#include <stdlib.h>

/*
Funtion that gets device information:
 - number of available GPUs
 - max number of threads per block
*/

deviceInfo_t* available_gpus;

void device_information (int num_devices) {
    
    // int num_devices;
    // cudaGetDeviceCount(&num_devices);
    // available_gpus = new DeviceInfo[num_devices];
    // make sure the array is allocated before calling the function
    
    for (int i=0; i<num_devices; i++) {
        cudaSetDevice(i);
        cudaDeviceProp gpu;
        cudaGetDeviceProperties(&gpu, 0);
        DeviceInfo dev;
        dev.name = gpu.name;
        dev.clockRate = gpu.clockRate;
        dev.maxThreadsperBlock = gpu.maxThreadsPerBlock;
        dev.maxBlocksperMulti = gpu.maxBlocksPerMultiProcessor;

        available_gpus[i] = dev;
    }
}

/*
Function that performs the index extraction
*/
__global__ void find_nearest_points (station_t * stationArr, float * lats, float * lons, int numStations, int num_lons_lats) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numStations) {
        float min_distance = 999;
        int min_index = -1;
        
        station_t * curr_station = &stationArr[id];
        float st_lat = (curr_station->latll + curr_station->latur) / 2.0;
        float st_lon = (curr_station->lonll + curr_station->lonur) / 2.0;

        for (int i=0; i<numberOfPoints; i++) {
            float lat = lats[i];
            float lon = lons[i]; 
            float distance = sqrt(pow((st_lat - lat), 2) + pow((st_lon - lon), 2));

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
void index_extraction (station_t & stationArr, float* lats, float* lons, int numStations, int numberOfPoints) { 
    
    // get useful device properties
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    available_gpus = new DeviceInfo[num_devices];
    cudaSetDevice(0);
    int max_threads = available_gpus[0].maxThreadsPerBlock;
    int max_blocks = available_gpus[0].maxBlocksPerMulti;

    int num_threads_to_use = max_threads / 2;
    int num_blocks_to_use = max_blocks / 2;

    // make device copies of host params
    station_t* d_stationArr;
    float* d_lats, d_lons;
    

    // allocate each of the arguments to the GPU
    if (cudaMalloc(&d_stationArr, sizeof(station_t) * numStations) != cudaSuccess) {
        std::cout << "stationArr could not be allocated to GPU" << std::endl;
        delete[] available_gpus;
        return;
    }
    if (cudaMalloc(&d_lats, sizeof(float) * numberOfPoints) != cudaSuccess) {
        std::cout << "Grib lats could not be allocated to the GPU" << std::endl;
        cudaFree(d_stationArr);
        delete[] available_gpus;
        return;
    }
    if (cudaMalloc(&d_lons, sizeof(float) * numberOfPoints) != cudaSuccess) {
        std::cout << "Grib lons could not be allocated to the GPU" << std::endl;
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        delete[] available_gpus;
    }

    // copy each of the arguments over to the gpu
    if (cudaMemcpy(d_stationArr, stationArr, sizeof(station_t) * numStations, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The stationArr could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        cudaFree(d_lons);
        delete[] available_gpus;
        return;
    }
    if (cudaMemcpy(d_lats, lats, sizeof(float) * numberOfPoints, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The grib lats could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        cudaFree(d_lons);
        delete[] available_gpus;
        return;
    }
    if (cudaMemcpy(d_lons, lons, sizeof(float) * numberOfPoints, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The grib lons could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        cudaFree(d_lons);
        delete[] available_gpus;
        return;
    }
    
    // call the kernel
    find_nearest_points <<< num_blocks_to_use, num_threads_to_use >>> (d_stationArr, d_lats, d_lons, numStations, numberOfPoints);
    
    // copy the elements from the GPU back over to the host
    if (cudaMemcpy(stationArr, d_stationArr, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_stationArr);
        cudaFree(d_lats);
        cudaFree(d_lons);
        delete[] available_gpus;
        return;
    }

    // release
    cudaFree(d_stationArr);
    cudaFree(d_lats);
    cudaFree(d_lons);
    delete[] available_gpus;


}
