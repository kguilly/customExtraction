#include "decompress_cuda.h"
#include "iostream"

/*
Funtion that gets device information:
 - number of available GPUs
 - max number of threads per block
*/
struct DeviceInfo {
    std::string name;
    int clockRate;
    int maxBlocksperMulti;
    int maxThreadsperBlock;
    size_t free_mem;
    size_t total_mem;
};
DeviceInfo* available_gpus;

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
__global__ void find_nearest_points () {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
}

/*
Function that orchestrates the index extraction
*/
void index_extraction (Station & stationArr, float lats, float lons) { 
    // just need the lats and lons of the stations

    // allocate each of the arguments to the GPU

    // copy each of the arguments over to the gpu

    // call the kernel

    // copy the elements from the GPU back over to the host

    // copy the nearest points over to the stationArr

}
