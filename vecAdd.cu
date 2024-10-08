#include <stdio.h>
#include "timer.h"

__host__ __device__ float f(float a, float b) {
    return a + b;
}

void vecadd_cpu(float* x, float* y, float* z, int N) {
    for(unsigned int i = 0; i < N; ++i) {
        z[i] = f(x[i], y[i]);
    }
}

__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < N) {
        z[i] = f(x[i], y[i]);
    }
}

void vecadd_gpu(float* x, float* y, float* z, int N) {

    // Allocate GPU memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**) &x_d, N*sizeof(float));
    cudaMalloc((void**) &y_d, N*sizeof(float));
    cudaMalloc((void**) &z_d, N*sizeof(float));

    // Copy data to GPU memory
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform computation on GPU
    const unsigned int numThresaPerBlock = 512;
    const unsigned int numBlocks = (N + numThresaPerBlock - 1) / numThresaPerBlock;
    vecadd_kernel<<<numBlocks, numThresaPerBlock>>>(x_d, y_d, z_d, N);

    // Copy data from GPU memory
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}


int main(int argc, char**argv){
    cudaDeviceSynchronize();
    Timer timer;

    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 25);
    float* x = (float*) malloc(N*sizeof(float));
    float* y = (float*) malloc(N*sizeof(float));
    float* z = (float*) malloc(N*sizeof(float));
    for (unsigned int i=0; i<N; ++i){
        x[i] = rand();
        y[i] = rand();
    }

    startTime(&timer);
    vecadd_cpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    startTime(&timer);
    vecadd_gpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    free(x);
    free(y);
    free(z);
}