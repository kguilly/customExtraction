#include "test_kernel.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <ctime>
#include <cuda_runtime.h>

__global__
void add_ints(int* a, int* b, int count){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < count) {
    a[id] += b[id];
  }
}

void the_kernel(int* x, int* y, int N){
  // Make device versions of the host variables
  int *d_x, *d_y;
  if (cudaMalloc(&d_x, sizeof(int)*N) != cudaSuccess){
  std::cout << "d_x could not be alloced" << std::endl;
  return;
  }

  if (cudaMalloc(&d_y, sizeof(int)*N) != cudaSuccess){
  std::cout << "d_y could not be alloced" << std::endl;
  cudaFree(d_x);
  return;
  }

  // Now copy the values over to the GPU
  if (cudaMemcpy(d_x, x, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess){
  std::cout << "Could not copy d_x" << std::endl;
  cudaFree(d_x);
  cudaFree(d_y);
  return;
  }

  if (cudaMemcpy(d_y, y, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess){
  std::cout << "Could not copy d_y" << std::endl;
  cudaFree(d_x);
  cudaFree(d_y);
  return;
  }

  add_ints<<<N / 256 + 1, 256 >>>(d_x, d_y, N);

  if (cudaMemcpy(x, d_x, sizeof(int) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
      delete[] x;
      delete[] y;  
      cudaFree(d_x);
      cudaFree(d_y);
      std::cout << "The stuff could not be copied back to the host" << std::endl;
      return;
  }

  cudaFree(d_x);
  cudaFree(d_y);
}