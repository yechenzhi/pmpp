#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 1024
#define NUM_BINS 256
#define COARSE_FACTOR 8

void histogram_cpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    for(unsigned int i = 0; i < width*height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

__global__ void histogram_kenel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height){

    __shared__ unsigned int bins_s[NUM_BINS];
    if (threadIdx.x < NUM_BINS){
        bins_s[threadIdx.x] = 0;
    }
    __syncthreads();
    unsigned int b =  blockIdx.x * blockDim.x * COARSE_FACTOR;
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c){
        unsigned int i = b + c * BLOCK_DIM + threadIdx.x;
        if(i < width*height){
            unsigned char b = image[i];
            atomicAdd(&bins_s[b], 1);
        }
        __syncthreads();
    }

    if (threadIdx.x < NUM_BINS && bins_s[threadIdx.x] > 0){
        atomicAdd(&bins[threadIdx.x], bins_s[threadIdx.x]);
    }
}

void histogram_gpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *image_d;
    unsigned int *bins_d;
    cudaMalloc((void**) &image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &bins_d, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // calll kernel
    startTime(&timer);
    // const unsigned int numThresaPerBlock = 512;
    // const unsigned int numBlocks = (N + numThresaPerBlock - 1) / numThresaPerBlock;
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numElementsPerBlock = numThreadsPerBlock*COARSE_FACTOR;
    unsigned int numBlocks = (width*height + numElementsPerBlock - 1)/numElementsPerBlock;
    histogram_kenel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(bins, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(image_d);
    cudaFree(bins_d);
}

int main(int argc, char** argv) {
    unsigned int width = 1024;
    unsigned int height = 1024*1024;

    // Allocate memory for RGB and gray images
    unsigned char* image = (unsigned char*) malloc(width*height * sizeof(unsigned char));
    unsigned int* bins_cpu = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
    unsigned int* bins_gpu = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));

    // Initialize RGB arrays with some values
    for (unsigned int i = 0; i < width*height; ++i) {
        image[i] = static_cast<unsigned char>(rand() % 256);
    }

    Timer timer;

    // CPU version
    startTime(&timer);
    histogram_cpu(image, bins_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU histogram", GREEN);

    // GPU version
    startTime(&timer);
    histogram_gpu(image, bins_gpu, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU histogram", GREEN);

    // Verify results
    float maxError = 0.0f;
    for (unsigned int i = 0; i < NUM_BINS; ++i) {
        float error = fabs(bins_cpu[i] - bins_gpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    printf("Max error: %f\n", maxError);
    printf("First 5 elements (CPU): %u %u %u %u %u\n", bins_cpu[0], bins_cpu[1], bins_cpu[2], bins_cpu[3], bins_cpu[4]);
    printf("First 5 elements (GPU): %u %u %u %u %u\n", bins_gpu[0], bins_gpu[1], bins_gpu[2], bins_gpu[3], bins_gpu[4]);
    printf("Last 5 elements (CPU): %u %u %u %u %u\n", bins_cpu[255-5], bins_cpu[255-4], bins_cpu[255-3], bins_cpu[255-2], bins_cpu[255-1]);
    printf("Last 5 elements (GPU): %u %u %u %u %u\n", bins_gpu[255-5], bins_gpu[255-4], bins_gpu[255-3], bins_gpu[255-2], bins_gpu[255-1]);

    // Free memory
    free(image);
    free(bins_cpu);
    free(bins_gpu);
    return 0;
}