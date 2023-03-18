#ifndef TEST_KERNEL_H_
#define TEST_KERNEL_H_

#include <cuda_runtime.h>

void the_kernel(int*x, int*y, int N);
__global__ void add_ints(int*a, int*b, int count);

#endif // TEST_KERNEL_H_
