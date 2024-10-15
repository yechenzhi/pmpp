#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 8

// CPU version of scan
void scan_cpu(float* input, float* output, unsigned int N) {
    output[0] = input[0];
    for(unsigned int i = 1; i < N; ++i){
        output[i] = input[i] + output[i-1];
    }
}

__global__ void scan_kernel(float* input, float* output, float* partialSums, unsigned int N){
    unsigned int bSegment = blockIdx.x*BLOCK_DIM*COARSE_FACTOR;

    __shared__ float buffer_s[BLOCK_DIM*COARSE_FACTOR];
    for(unsigned int c=0; c < COARSE_FACTOR; ++c){
        buffer_s[threadIdx.x+c*BLOCK_DIM] = input[bSegment+c*BLOCK_DIM+threadIdx.x];
    }
    __syncthreads();

    // Thread scan
    unsigned int tSegment = COARSE_FACTOR*threadIdx.x; 
    for(unsigned int c=1; c< COARSE_FACTOR; ++c){
        buffer_s[tSegment+c] += buffer_s[tSegment+c-1];
    }
    __syncthreads();

    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;
    inBuffer_s[threadIdx.x] = buffer_s[tSegment + COARSE_FACTOR - 1];
    __syncthreads();

    for(unsigned int stride = 1; stride <= BLOCK_DIM/2;  stride *= 2){
        if(threadIdx.x >= stride){
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else{
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float* tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }
    
    if(threadIdx.x > 0){
        for(unsigned int c=0; c<COARSE_FACTOR; ++c){
            buffer_s[tSegment+c] += inBuffer_s[threadIdx.x-1];
        }
    }
    if(threadIdx.x == BLOCK_DIM - 1){
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x]; // note here.
    }
    __syncthreads();
    for(unsigned int c=0; c< COARSE_FACTOR; ++c){
        output[bSegment+c*BLOCK_DIM+threadIdx.x] = buffer_s[c*BLOCK_DIM+threadIdx.x];
    }
}

__global__ void add_kernel(float* output, float* partialSums, unsigned int N){
    unsigned int bSegment = blockIdx.x*blockDim.x*COARSE_FACTOR; 
    if(blockIdx.x > 0){
        for(unsigned int c=0; c< COARSE_FACTOR; ++c){
            output[bSegment+c*BLOCK_DIM+threadIdx.x] += partialSums[blockIdx.x-1];
        }
    }
}

void scan_gpu_d(float* input_d, float* output_d, unsigned int N){
    Timer timer; 

    // configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = COARSE_FACTOR*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;

    // Allocate partial sums 
    startTime(&timer);
    float *partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(float));
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

void scan_gpu(float* input, float* output, unsigned int N){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *input_d, *output_d;
    cudaMalloc((void**) &input_d, N*sizeof(float));
    cudaMalloc((void**) &output_d, N*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time", BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
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
    cudaMemcpy(output, output_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(input_d);
    cudaFree(output_d);
}

int main(int argc, char** argv) {
    int N = BLOCK_DIM*BLOCK_DIM*10;
    // int N = BLOCK_DIM*100;

    // Allocate memory for RGB and gray images
    float* input = (float*) malloc(N * sizeof(float));
    float* output_cpu = (float*) malloc(N * sizeof(float));
    float* output_gpu = (float*) malloc(N * sizeof(float));

    // Initialize RGB arrays with some values
    for (unsigned int i = 0; i < N; ++i) {
        input[i] = (float)rand() / (float)RAND_MAX;
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
    float maxError = 0.0f;
    for (unsigned int i = 0; i < N; ++i) {
        float error = fabs(output_cpu[i] - output_gpu[i]);
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