#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 1024

// CPU version of scan
void scan_cpu(double* input, double* output, unsigned int N) {
    output[0] = input[0];
    for(unsigned int i = 1; i < N; ++i){
        output[i] = input[i] + output[i-1];
    }
}

__global__ void scan_kernel(double* input, double* output, double* partialSums, unsigned int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ double buffer_s[BLOCK_DIM];
    buffer_s[threadIdx.x] = input[i];
    __syncthreads();

    for(unsigned int stride = 1; stride <= BLOCK_DIM/2;  stride *= 2){
        double v;
        if(threadIdx.x >= stride){
            v = buffer_s[threadIdx.x-stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            buffer_s[threadIdx.x] += v;
        }
        __syncthreads();
    }
    if(threadIdx.x == BLOCK_DIM - 1){
        partialSums[blockIdx.x] = buffer_s[threadIdx.x];
    }
    output[i] = buffer_s[threadIdx.x];
}

__global__ void add_kernel(double* output, double* partialSums, unsigned int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
    if(blockIdx.x > 0){
        output[i] += partialSums[blockIdx.x-1];
    }
}

void scan_gpu_d(double* input_d, double* output_d, unsigned int N){
    Timer timer; 

    // configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;

    // Allocate partial sums 
    startTime(&timer);
    double *partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(double));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "partial sums allocation time",BLUE);

    // call kernel 
    startTime(&timer);
    scan_kernel<<<numBlocks, numThreadsPerBlock>>> (input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "kernel time", GREEN);

    // scan partial sums then add 
    if(numBlocks > 1){
        // scan partial sums
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);
        // add scanned sums
        add_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d, partialSums_d, N);
    }
    
    startTime(&timer);
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "deallocation time", BLUE);
}

void scan_gpu(double* input, double* output, unsigned int N){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    double *input_d, *output_d;
    cudaMalloc((void**) &input_d, N*sizeof(double));
    cudaMalloc((void**) &output_d, N*sizeof(double));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time", BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(input_d, input, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time", BLUE);

    // Perform computation on GPU
    startTime(&timer);
    scan_gpu_d(input_d, output_d, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "total Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(output, output_d, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(input_d);
    cudaFree(output_d);
}

int main(int argc, char** argv) {
    int N = BLOCK_DIM*BLOCK_DIM*100;
    // int N = BLOCK_DIM*100;

    // Allocate memory for RGB and gray images
    double* input = (double*) malloc(N * sizeof(double));
    double* output_cpu = (double*) malloc(N * sizeof(double));
    double* output_gpu = (double*) malloc(N * sizeof(double));

    // Initialize RGB arrays with some values
    for (unsigned int i = 0; i < N; ++i) {
        input[i] = (double)rand() / (double)RAND_MAX;
    }
    

    Timer timer;

    // CPU version
    startTime(&timer);
    scan_cpu(input, output_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU scan", GREEN);

    // GPU version
    startTime(&timer);
    scan_gpu(input, output_gpu, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU scan", GREEN);

    // Verify results
    double maxError = 0.0f;
    for (unsigned int i = 0; i < N; ++i) {
        double error = fabs(output_cpu[i] - output_gpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    printf("Max error: %f\n", maxError);
    printf("First 5 elements (CPU): %f %f %f %f %f\n", output_cpu[0], output_cpu[1], output_cpu[2], output_cpu[3], output_cpu[4]);
    printf("First 5 elements (GPU): %f %f %f %f %f\n", output_gpu[0], output_gpu[1], output_gpu[2], output_gpu[3], output_gpu[4]);
    printf("Last 5 elements (CPU): %f %f %f %f %f\n", output_cpu[N-5], output_cpu[N-4], output_cpu[N-3], output_cpu[N-2], output_cpu[N-1]);
    printf("Last 5 elements (GPU): %f %f %f %f %f\n", output_gpu[N-5], output_gpu[N-4], output_gpu[N-3], output_gpu[N-2], output_gpu[N-1]);
    // Free memory
    free(input);
    free(output_cpu);
    free(output_gpu);
    return 0;
}