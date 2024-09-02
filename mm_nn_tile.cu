#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32

// CPU version of matrix multiplication
void mm_cpu(float* A, float* B, float* C, unsigned int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum += A[row * N + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

__global__ void mm_kenel_tile(float* A, float* B, float* C, unsigned int N){
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    // TODO: N/TILE_DIM is this right? 
    for(unsigned int tile=0; tile < N/TILE_DIM; ++tile){
        A_s[threadIdx.y][threadIdx.x] = A[row*N+tile*TILE_DIM+threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM+threadIdx.y)*N+col];
        __syncthreads();

        for(unsigned int i = 0; i< TILE_DIM; ++i){
            sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
        }
         __syncthreads();
    }
    C[row*N+col] = sum;
}

void mm_gpu(float* A, float* B, float* C, unsigned int N){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, N*N*sizeof(float));
    cudaMalloc((void**) &B_d, N*N*sizeof(float));
    cudaMalloc((void**) &C_d, N*N*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // Perform computation on GPU
    startTime(&timer);
    // const unsigned int numThresaPerBlock = 512;
    // const unsigned int numBlocks = (N + numThresaPerBlock - 1) / numThresaPerBlock;
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(N + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    mm_kenel_tile<<<numBlocks, numThreadsPerBlock>>>(A_d,B_d,C_d,N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",BLUE);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(C, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char** argv) {
    unsigned int N = 1 << 10;
    unsigned int size = N * N;

    // Allocate memory for RGB and gray images
    float* A = (float*) malloc(size * sizeof(float));
    float* B = (float*) malloc(size * sizeof(float));
    float* C_cpu = (float*) malloc(size * sizeof(float));
    float* C_gpu = (float*) malloc(size * sizeof(float));

    // Initialize RGB arrays with some values
    for (unsigned int i = 0; i < size; ++i) {
        A[i] = rand() % 256;
        B[i] = rand() % 256;
    }

    Timer timer;

    // CPU version
    startTime(&timer);
    mm_cpu(A,B,C_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU RGB to Gray", GREEN);

    // GPU version
    startTime(&timer);
    mm_gpu(A,B,C_gpu, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU RGB to Gray", GREEN);

    // Verify results
    float maxError = 0.0f;
    for (unsigned int i = 0; i < size; ++i) {
        float error = fabs(C_cpu[i] - C_gpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    printf("Max error: %f\n", maxError);

    // Free memory
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    return 0;
}