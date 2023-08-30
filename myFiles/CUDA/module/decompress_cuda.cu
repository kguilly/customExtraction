
#include "iostream"
#include <stdlib.h>
#include "eccodes.h"
#include <cuda_runtime.h>

#include "decompress_cuda.h"
#include "decompress_funcs.h"
#include "shared_objs.h"
#define MAX_VAL_LEN 1024

// cudaExternalSemaphore_t * values_semaphores; // protects the station array from being written
//                                      // to by multiple threads

// void initCudaSem(int numStations){
//     if (cudaMallocManaged(&values_semaphores, numStations * sizeof(cudaExternalSemaphore_t)) != cudaSuccess) {
//         std::cout << "semaphores could not be allocated" << std::endl;
//     }
//     for (int i = 0; i < numStations; i++) {
//         cudaExternalSemaphore_t extSemaphore;
//         CUDA_CHECK(cudaExternalSemaphoreCreate(&extSemaphore, NULL, NULL));
//         if ()
//         values_semaphores[i] = extSemaphore;
//     }
// }

// void destroyCudaSem(int numStations) {

// }
__device__ station_t * d_stationArr; // global pointer array. Same as stationArr, except will go on


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
    station_t* d_stationArr_findNearest;
    double* d_lats;
    double* d_lons;
    

    // allocate each of the arguments to the GPU
    if (cudaMalloc(&d_stationArr_findNearest, sizeof(station_t) * numStations) != cudaSuccess) {
        std::cout << "stationArr could not be allocated to GPU" << std::endl;
        return;
    }
    if (cudaMalloc(&d_lats, sizeof(double) * numberOfPoints) != cudaSuccess) {
        std::cout << "Grib lats could not be allocated to the GPU" << std::endl;
        cudaFree(d_stationArr_findNearest);
        return;
    }
    if (cudaMalloc(&d_lons, sizeof(double) * numberOfPoints) != cudaSuccess) {
        std::cout << "Grib lons could not be allocated to the GPU" << std::endl;
        cudaFree(d_stationArr_findNearest);
        cudaFree(d_lats);
        return;
    }

    // copy each of the arguments over to the gpu
    if (cudaMemcpy(d_stationArr_findNearest, stationArr, sizeof(station_t) * numStations, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The stationArr could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr_findNearest);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    if (cudaMemcpy(d_lats, lats, sizeof(double) * numberOfPoints, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The grib lats could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr_findNearest);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    if (cudaMemcpy(d_lons, lons, sizeof(double) * numberOfPoints, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The grib lons could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr_findNearest);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    
    // call the kernel
    find_nearest_points <<< num_blocks_to_use, num_threads_to_use >>> (d_stationArr_findNearest, d_lats, d_lons, numStations, numberOfPoints);
    // wait for em all to finish
    cudaDeviceSynchronize();

    // copy the elements from the GPU back over to the host
    if (cudaMemcpy(stationArr, d_stationArr_findNearest, sizeof(station_t) * numStations, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "cuda FAILED" << std::endl;
        cudaFree(d_stationArr_findNearest);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    std::cout << "cuda success" << std::endl;
    // release
    cudaFree(d_stationArr_findNearest);
    cudaFree(d_lats);
    cudaFree(d_lons);

}


/*
CUDA kernel that indexes the appropriate values to each station
*/
__global__ void get_station_values(int hour, int currParamIdx, int numStations, int numberOfPoints, double* grib_values){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numStations) {
        d_stationArr[idx].values[hour][currParamIdx] = grib_values[d_stationArr[idx].closestPoint];
    }
}

/*
CPU threading function 
*/
void * read_grib_data(void *arg) {
    // Grab the data from the thread arg
    threadArgs_t * this_arg = (threadArgs_t*)arg;
    FILE * f = (*this_arg).f;
    const char * full_path = (*this_arg).pathName;
    int threadIndex = (*this_arg).threadIndex;
    // const char * hour = (*this_arg).hour;
    // const char * strCurrentDay = (*this_arg).strCurrentDay;
    bool first_hour_flag = (*this_arg).first_hour_flag;
    bool last_hour_flag = (*this_arg).last_hour_flag;
    bool* blnParamArr = (*this_arg).blnParamArr;
    sem_t *barrier = this_arg->barrier;
    size_t numStations = this_arg->numStations;
    station_t * stationArr;
    deviceInfo_t gpu = this_arg->gpu; 

    // find number of threads and blocks to use
    int num_blocks_to_use = gpu.maxBlocksperMulti / 2;
    int num_threads_to_use = gpu.maxThreadsperBlock / 2;

    // if first hour
        // copy over new station array

    // if last hour
        // copy station array from device back to host....
        // or just copy the values array over either way

    // put a sem right here so they can do their business before getting started 
    if (threadIndex != 0) {
        sem_wait(barrier);
        sem_post(barrier);
    }
    if (first_hour_flag) {
        // if this is the first hour, grab the station array and then copy it over
        stationArr = (*this_arg).stationArr;
        numStations = (*this_arg).numStations;
        if (cudaMalloc(&d_stationArr, sizeof(station_t) * numStations) != cudaSuccess) {
            std::cout << "Mem forstationArr could not be allocated to GPU" << std::endl;
            exit(1);
        }
        if (cudaMemcpy(d_stationArr, stationArr, numStations * sizeof(station_t), cudaMemcpyHostToDevice) != cudaSuccess){
            std::cout << "stationArr could not be placed on GPU \n";
            cudaFree(d_stationArr);
            exit(1);
        }
    }

    // once the first hour has done its dirty work, let the rest of them go
    if (threadIndex == 0) sem_post(barrier);
    std::cout << full_path << std::endl;
    // Now open the grib file and extract the values array
    try {
        f = fopen(full_path, "rb");
        if (!f) throw(full_path);
    }
    catch (std::string file) {
        std::cout << "Error: when reading GRIB data, could not open file " << file << std::endl;
        if (last_hour_flag) {
            if (cudaMemcpy(stationArr, d_stationArr, numStations * sizeof(station_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::cout << "stationArr could not be copied back to the host" << std::endl;
                cudaFree(d_stationArr);
                exit(1);
            }
        }
        return nullptr;
    }

    codes_handle * h = NULL; // use to unpack each layer of the file
    const double missing = 1.0e36;        // placeholder for when the value cannot be found in the grib file
    int msg_count = 0;  // KEY: will match with the layer of the passed parameters
    int err = 0;
    long numberOfPoints=0;
    double *grib_values, *grib_lats, *grib_lons;
    std::string name_space = "parameter";
    int currParamIdx = 0;


    while ((h=codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err)) != NULL) {
        msg_count++;

        if (blnParamArr[msg_count] == false){
            codes_handle_delete(h);
            continue;
        }

        CODES_CHECK(codes_get_long(h, "numberOfPoints", &numberOfPoints), 0);
        CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);
        grib_lats = (double*)malloc(numberOfPoints * sizeof(double));
        if(!grib_lats){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
            exit(0);
        }
        grib_lons = (double*)malloc(numberOfPoints * sizeof(double));
        if (!grib_lons){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
            std::free(grib_lats);
            exit(0);
        }
        grib_values = (double*)malloc(numberOfPoints * sizeof(double));
        if(!grib_values){
            fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(numberOfPoints * sizeof(double)));
            std::free(grib_lats);
            std::free(grib_lons);
            exit(0);
        }
        CODES_CHECK(codes_grib_get_data(h, grib_lats, grib_lons, grib_values), 0);

        // now we need to call the kernel with ?stationarr?, numstations, numberOfPoints, and grib_values
        double * d_grib_values;
        if (cudaMalloc(&d_grib_values, sizeof(double) * numberOfPoints) != cudaSuccess) {
            std::cout << "mem for values could not be allocated for thread " << threadIndex << std::endl;
            return nullptr;
        }
        if (cudaMemcpy(d_grib_values, grib_values, sizeof(double) * numberOfPoints, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cout << "values could not be put on gpu for thread " << threadIndex << std::endl;
            cudaFree(d_grib_values);
            return nullptr;
        }
        // allocate and copy over the grib_values array
        int hour = threadIndex;
        
        get_station_values <<< num_blocks_to_use, num_threads_to_use >>> (hour, currParamIdx, numStations, numberOfPoints, grib_values);
        // update currParamIdx
        currParamIdx++;
        
        cudaFree(d_grib_values);
        std::free(grib_lats);
        std::free(grib_lons);
        std::free(grib_values);

        codes_handle_delete(h);

    }

    if (last_hour_flag) {
        // block the CPU until all other threads have finished
        cudaDeviceSynchronize();
        // copy station array from device back to host
        if (cudaMemcpy(stationArr, d_stationArr, numStations * sizeof(station_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cout << "stationArr could not be copied back to the host" << std::endl;
            cudaFree(d_stationArr);
            exit(1);
        }
    }
    return nullptr;    
}