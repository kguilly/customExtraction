// this test will add the elements of two arrays with a million elemetns each
// I also want to test how to get information about the GPU
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{

  int N = 1<<20;
  float *x, *y;

  /////////////////////////////////////////////////////////////////////
  // Get device properties

  int num_devices;
  cudaGetDeviceCount(&num_devices);
  for(int i=0; i < num_devices; i++){
    cudaDeviceProp prop;
    cudaGetDeviceProperties_v2(&prop, i);
    std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
    
    size_t free_mem, total_mem;
    cudaSetDevice(i);
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Total Memory: " << total_mem << "\t Free Memory: " << free_mem << std::endl;

  }
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
