#include <ctime>
#include <cuda_runtime.h>

using namespace std;

__global__ void AddInts(int *a, int* b, int count);

int main() {
    srand(time(NULL));
    int count = 100;
    int *h_a = new int[count];
    int *h_b = new int[count];

    for (int i=0; i<count; i++) {
        h_a[i] = rand() % 1000;
        h_b[i] = rand() % 1000;
    }

    // init device variables
    int *d_a, *d_b;

    // allocate space
    if (cudaMalloc(&d_a, sizeof(int) * count) != cudaSuccess) {
        exit(0);
    }
    if (cudaMalloc(&d_b, sizeof(int) * count) != cudaSuccess) {
        cudaFree(d_a);
        exit(0);
    }

    // copy values into device arrays
    if (cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        exit(0);
    }
    if (cudaMemCpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        exit(0);
    }

    // launch the kernel
    AddInts <<< count / 256 + 1, 256 >>> (d_a, d_b, count);

    // copy the memory back to the host
    if (cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost) != cudaSuccess) {
        delete [] h_a;
        delete [] h_b;
        cudaFree(d_a);
        cudaFree(d_b);
        exit(0);
    }
    delete [] h_a;
    delete [] h_b;
}

__global__ void AddInts(int *a, int* b, int count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        a[id] += b[id];
    }
}