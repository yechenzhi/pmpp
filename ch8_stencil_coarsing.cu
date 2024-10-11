#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 32
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
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
    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = blockIdx.y*OUT_TILE_DIM+threadIdx.y-1;
    int k = blockIdx.x*OUT_TILE_DIM+threadIdx.x-1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    if(iStart >= 1 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N){
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart-1)*N*N+j*N+k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N){
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N+j*N+k];
    }

    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i){
        if(i+1>=0 && i+1<N && j+1>=0 && j+1<N && k+1>=0 && k+1<N){
            inNext_s[threadIdx.y][threadIdx.x] = in[(i+1)*N*N+j*N+k];
        }
        __syncthreads();
        if(i>=1 && i < N-1 && j>=1 && j < N-1 && k>=1 && k < N-1){
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1){
                out[i*N*N+j*N+k] = C0 * inCurr_s[threadIdx.y][threadIdx.x] +
                                   C1 * (inCurr_s[threadIdx.y][threadIdx.x-1] + inCurr_s[threadIdx.y][threadIdx.x+1] +
                                         inCurr_s[threadIdx.y-1][threadIdx.x-1] + inCurr_s[threadIdx.y+1][threadIdx.x+1] +
                                         inPrev_s[threadIdx.y][threadIdx.x] + 
                                         inNext_s[threadIdx.y][threadIdx.x]);

            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
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
    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 numBlocks((N + OUT_TILE_DIM - 1)/OUT_TILE_DIM,(N + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
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