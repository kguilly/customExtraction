// this test will add the elements of two arrays with a million elemetns each
// I also want to test how to get information about the GPU
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <ctime>
// Kernel function to add the elements of two arrays

__global__ void AddInts(int*, int*, int);

int main(void)
{

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
    std::cout << "Total Memory: " << total_mem << "\t Free Memory: " << free_mem << std::endl;

  }

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

  // Make device versions of the host variables
  int *d_x, *d_y;
  if (cudaMalloc(&d_x, sizeof(int)*N) != cudaSuccess){
    std::cout << "d_x could not be alloced" << std::endl;
    return 0;
  }

  if (cudaMalloc(&d_y, sizeof(int)*N) != cudaSuccess){
    std::cout << "d_y could not be alloced" << std::endl;
    cudaFree(d_x);
    return 0;
  }

  // Now copy the values over to the GPU
  if (cudaMemcpy(d_x, h_x, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess){
    std::cout << "Could not copy d_x" << std::endl;
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
  }

  if (cudaMemcpy(d_y, h_y, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess){
    std::cout << "Could not copy d_y" << std::endl;
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
  }

  AddInts<<<N / 256 + 1, 256 >>>(d_x, d_y, N);

  if (cudaMemcpy(h_x, d_x, sizeof(int) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
    delete[] h_x;
    delete[] h_y;  
    cudaFree(d_x);
    cudaFree(d_y);
    std::cout << "The stuff could not be copied back to the host" << std::endl;
    return 0;
  }

  // check to see that it did everything correctly
  std::cout << "\n After Addition: " << std::endl;
  for (int i=0; i<5; i++) {
    std::cout << h_x[i] << std::endl;
  }

  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_x;
  delete[] h_y;
  
  return 0;
}

__global__ void AddInts(int* a, int* b, int count){
  
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < count) {
    a[id] += b[id];
  }
  
}