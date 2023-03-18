#include "test_kernel.h"
#include <iostream>
#include <stdlib.h>


int main(){
      srand(time(NULL));

  /////////////////////////////////////////////////////////////////////
  // Get device properties
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  for(int i=0; i < num_devices; i++){
    cudaDeviceProp prop;
    cudaGetDeviceProperties_v2(&prop, i);
    std::cout << "\nDevice " << i << ": " << prop.name << "\n - Clock Rate: ";
    std::cout << prop.clockRate << "\n - Compute Mode: " << prop.computeMode;
    std::cout << "\n - MaxGridSize: " << prop.maxGridSize << "\n - Max Threads Per Block: ";
    std::cout << prop.maxThreadsPerBlock << "\n - Max Threads Dim: " << prop.maxThreadsDim << std::endl;
    std::cout << " - UUID: " << /*prop.uuid*/ "\n - Warp Size: " << prop.warpSize << std::endl;
    std::cout << " - Max Blocks per multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    
    size_t free_mem, total_mem;
    cudaSetDevice(i);
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\nTotal Memory: " << total_mem << "\t Free Memory: " << free_mem << std::endl;

  }
  std::cout << " " << std::endl;
  
    int N = 1<<20; // I think thats a binary 1 shifted left 20 times
    int* h_x = new int[N];
    int* h_y = new int[N];
    //make host version of the arrays
    for (int i=0; i<N; i++) {
    h_x[i] = rand() % 1000;
    h_y[i] = rand() % 1000;
    }

    for (int i=0; i<5; i++) {
    std::cout << h_x[i] << " " << h_y[i] << std::endl;
    }

    // call the cuda device kernel
    the_kernel(h_x, h_y, N);

    // print the result
    std::cout << "\n After Addition: " << std::endl;
    for (int i=0; i<5; i++) {
        std::cout << h_x[i] << std::endl;
    }

    delete [] h_x;
    delete [] h_y;
}