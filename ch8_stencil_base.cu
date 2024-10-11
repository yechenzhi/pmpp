#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 8 
#define C0 0.2
#define C1 0.8 

// CPU version of matrix multiplication
void stencil_cpu(float* in, float* out, unsigned int N) {
    for (unsigned int i = 1; i < N-1; ++i) {
        for (unsigned int j = 1; j < N-1; ++j) {
            for (unsigned int k = 1; k < N-1; ++k) {
                out[i*N*N + j*N + k] = C0 * in[i*N*N + j*N + k] +
                                       C1 * (in[i*N*N + j*N + k-1] + in[i*N*N + j*N + k+1] +
                                             in[i*N*N + (j-1)*N + k] + in[i*N*N + (j+1)*N + k] +
                                             in[(i-1)*N*N + j*N + k] + in[(i+1)*N*N + j*N + k]);
            }
        }
    }
}

__global__ void stencil_kenel(float* in, float* out, unsigned int N){
    int i = blockIdx.z*blockDim.z+threadIdx.z;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    int k = blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=1 && i < N-1 && j>=1 && j < N-1 && k>=1 && k < N-1){
        out[i*N*N+j*N+k] = C0 * in[i*N*N+j*N+k] +
                           C1 * (in[i*N*N+j*N+k-1] + in[i*N*N+j*N+k+1] +
                                 in[i*N*N+(j-1)*N+k] + in[i*N*N+(j+1)*N+k] + 
                                 in[(i-1)*N*N+j*N+k] + in[(i+1)*N*N+j*N+k]);

    }

}

void stencil_gpu(float* in, float* out, unsigned int N){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *in_d, *out_d;
    cudaMalloc((void**) &in_d, N*N*N*sizeof(float));
    cudaMalloc((void**) &out_d, N*N*N*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(in_d, in, N*N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, N*N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // Perform computation on GPU
    startTime(&timer);
    // const unsigned int numThresaPerBlock = 512;
    // const unsigned int numBlocks = (N + numThresaPerBlock - 1) / numThresaPerBlock;
    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((N + BLOCK_DIM - 1)/BLOCK_DIM,(N + BLOCK_DIM - 1)/BLOCK_DIM, (N + BLOCK_DIM - 1)/BLOCK_DIM);
    stencil_kenel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(out, out_d, N*N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(in_d);
    cudaFree(out_d);
}

int main(int argc, char** argv) {
    unsigned int N = 512;
    unsigned int size = N * N * N;

    // Allocate memory for RGB and gray images
    float* in = (float*) malloc(size * sizeof(float));
    float* out_cpu = (float*) malloc(size * sizeof(float));
    float* out_gpu = (float*) malloc(size * sizeof(float));

    // Initialize RGB arrays with some values
    for (unsigned int i = 0; i < size; ++i) {
        in[i] = rand();
    }

    Timer timer;

    // CPU version
    startTime(&timer);
    stencil_cpu(in, out_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU RGB to Gray", GREEN);

    // GPU version
    startTime(&timer);
    stencil_gpu(in, out_gpu, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU RGB to Gray", GREEN);

    // Verify results
    float maxError = 0.0f;
    for (unsigned int i = 0; i < size; ++i) {
        float error = fabs(out_cpu[i] - out_gpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    printf("Max error: %f\n", maxError);

    // Free memory
    free(in);
    free(out_cpu);
    free(out_gpu);
    return 0;
}