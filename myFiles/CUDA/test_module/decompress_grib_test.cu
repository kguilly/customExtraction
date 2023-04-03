#include <iostream>
#include <stdlib.h>
#include "eccodes.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

const char * grib_file_path = "/media/kaleb/extraSpace/wrf/";
const char * output_file_path = "/home/kaleb/Desktop/cuda_4-3/";
std::vector<std::string> vctrDate = {"2020", "01", "01"};

typedef struct Station{
    const char* fips_code;
    float lat;
    float lon;
    double **values;
    int closestPoint;
} station_t;

station_t * stationArr;
int numStations;

void extract_indexes();
__global__ void cuda_find_nearest(station_t*, double*, double*, int, int);

int main() {
    station_t st_1;
    station_t st_2;
    st_1.lat = 30.2241;
    st_1.lon = 272.0198;

    st_2.lat = 29.7958;
    st_2.lon = 270.8229;

    station_t* stationArr = new station_t[2];
    numStations = 2;
    stationArr[0] = st_1;
    stationArr[1] = st_2;

    extract_indexes();

    for (int i=0; i<numStations; i++) {
        station_t st = stationArr[i];
        std::cout << "CL PT for st " << i << ": " << st.closestPoint << std::endl;
    }

    delete [] stationArr;

    return 0;
}

void extract_indexes(){

    FILE* f;
    // std::string file_name = grib_file_path + vctrDate.at(0) + "/" + \
    //                     vctrDate.at(0) + vctrDate.at(1) + vctrDate.at(2)
    const char * full_file_name = "/media/kaleb/extraSpace/wrf/2020/20200101/hrrr.20200101.00.00.grib2";

    try {
        f = fopen(full_file_name, "rb");
        if (!f) throw(full_file_name);
    }
    catch (std::string file) {
        std::cout << "could not open file: " << file << std::endl;
        return;
    }

    long num_points =0;
    const double missing = 1.23456e36;
    codes_handle * h = NULL;
    int err = 0;
    double *grib_lats, *grib_lons, *grib_values;
    h = codes_handle_new_from_file(0, f, PRODUCT_GRIB, &err);
    if (!h || err != 0) {
        std::cout << "could not make handle" << std::endl;
    }
    CODES_CHECK(codes_get_long(h, "numberOfPoints", &num_points), 0);
    CODES_CHECK(codes_set_double(h, "missingValue", missing), 0);

    grib_lats = (double*)malloc(num_points * sizeof(double));
    if(!grib_lats){
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(num_points * sizeof(double)));
        exit(0);
    }
    grib_lons = (double*)malloc(num_points * sizeof(double));
    if (!grib_lons){
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(num_points * sizeof(double)));
        std::free(grib_lats);
        exit(0);
    }
    grib_values = (double*)malloc(num_points * sizeof(double));
    if(!grib_values){
        fprintf(stderr, "Error: unable to allocate %ld bytes\n", (long)(num_points * sizeof(double)));
        std::free(grib_lats);
        std::free(grib_lons);
        exit(0);
    }
    CODES_CHECK(codes_grib_get_data(h, grib_lats, grib_lons, grib_values), 0);

    cudaSetDevice(0);
    int num_threads = 256;
    int num_blocks = 10;

    station_t* d_stationArr_np;
    double *d_lats, *d_lons;

    // allocate each of the arguments to the GPU
    if (cudaMalloc(&d_stationArr_np, sizeof(station_t) * numStations) != cudaSuccess) {
        std::cout << "stationArr could not be allocated to GPU" << std::endl;
        return;
    }
    if (cudaMalloc(&d_lats, sizeof(double) * num_points) != cudaSuccess) {
        std::cout << "Grib lats could not be allocated to the GPU" << std::endl;
        cudaFree(d_stationArr_np);
        return;
    }
    if (cudaMalloc(&d_lons, sizeof(double) * num_points) != cudaSuccess) {
        std::cout << "Grib lons could not be allocated to the GPU" << std::endl;
        cudaFree(d_stationArr_np);
        cudaFree(d_lats);
        return;
    }

    // copy each of the arguments over to the gpu
    if (cudaMemcpy(d_stationArr_np, stationArr, sizeof(station_t) * numStations, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The stationArr could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr_np);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    if (cudaMemcpy(d_lats, grib_lats, sizeof(double) * num_points, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The grib lats could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr_np);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    if (cudaMemcpy(d_lons, grib_lons, sizeof(double) * num_points, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "The grib lons could not be copied to the GPU" << std::endl;
        cudaFree(d_stationArr_np);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    
    // call the kernel
    cuda_find_nearest <<< num_blocks, num_threads >>> (d_stationArr_np, d_lats, d_lons, numStations, num_points);
    // wait for em all to finish
    cudaDeviceSynchronize();

    // copy the elements from the GPU back over to the host
    if (cudaMemcpy(stationArr, d_stationArr_np, sizeof(station_t) * numStations, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "cuda FAILED" << std::endl;
        cudaFree(d_stationArr_np);
        cudaFree(d_lats);
        cudaFree(d_lons);
        return;
    }
    std::cout << "cuda success" << std::endl;
    // release
    cudaFree(d_stationArr_np);
    cudaFree(d_lats);
    cudaFree(d_lons);

    std::free(grib_lats);
    std::free(grib_lons);
    std::free(grib_values);
}

__global__ void cuda_find_nearest(station_t * d_stationArr, double * d_lats, double * d_lons, int numStations, int num_points) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numStations) {
        double min_distance = 999;
        int min_index = -1;
        
        station_t * curr_station = &d_stationArr[id];
        double st_lat = curr_station->lat;
        double st_lon = curr_station->lon;

        for (int i=0; i<num_points; i++) {
            double lat = d_lats[i];
            double lon = d_lons[i]; 
            double distance = sqrt(pow((st_lat - lat), 2) + pow((st_lon - lon), 2));

            if (distance < min_distance) {
                min_distance = distance;
                min_index = i;
            }
        }
        curr_station->closestPoint = min_index;
    }
}
