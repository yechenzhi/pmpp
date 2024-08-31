#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLUR_SIZE 3

// CPU version of RGB to grayscale conversion
void blur_cpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height){
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            unsigned int sum = 0;
            for (int inRow = outRow - BLUR_SIZE; inRow <= outRow + BLUR_SIZE; ++inRow) {
                for (int inCol = outCol - BLUR_SIZE; inCol <= outCol + BLUR_SIZE; ++inCol) {
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        sum += image[inRow * width + inCol];
                    }
                }
            }
            blurred[outRow * width + outCol] = (unsigned char)(sum / ((2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1)));
        }
    }
}

__global__ void blur_kenel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height){
    int outRow = blockIdx.y*blockDim.y+threadIdx.y;
    int outCol = blockIdx.x*blockDim.x+threadIdx.x;

    if (outRow<height && outCol<width){
        unsigned int sum = 0;
        for(int inRow = outRow-BLUR_SIZE; inRow <= outRow+BLUR_SIZE; ++inRow){
            for(int inCol = outCol-BLUR_SIZE; inCol <= outCol+BLUR_SIZE; ++inCol){
                if(inRow>=0 && inRow<height && inCol>=0 && inCol<width){
                    sum += image[inRow*width+inCol];
                }
            }
        }
        blurred[outRow*width+outCol] = (unsigned char)(sum / ((2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)));
    }
}

void blur_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *image_d, *blurred_d;
    cudaMalloc((void**) &image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &blurred_d, width*height*sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // Perform computation on GPU
    startTime(&timer);
    // const unsigned int numThresaPerBlock = 512;
    // const unsigned int numBlocks = (N + numThresaPerBlock - 1) / numThresaPerBlock;
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    blur_kenel<<<numBlocks, numThreadsPerBlock>>>(image_d, blurred_d, width, height);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",BLUE);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(blurred, blurred_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(image_d);
    cudaFree(blurred_d);
}

int main(int argc, char** argv) {
    unsigned int width = 1 << 13;
    unsigned int height = 1 << 13;
    unsigned int size = width * height;

    // Allocate memory for RGB and gray images
    unsigned char* image = (unsigned char*) malloc(size * sizeof(unsigned char));
    unsigned char* blurred_cpu = (unsigned char*) malloc(size * sizeof(unsigned char));
    unsigned char* blurred_gpu = (unsigned char*) malloc(size * sizeof(unsigned char));

    // Initialize RGB arrays with some values
    for (unsigned int i = 0; i < size; ++i) {
        image[i] = rand() % 256;
    }

    Timer timer;

    // CPU version
    startTime(&timer);
    blur_cpu(image, blurred_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU RGB to Gray", GREEN);

    // GPU version
    startTime(&timer);
    blur_gpu(image, blurred_gpu, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU RGB to Gray", GREEN);

    // Verify results
    for (unsigned int i = 0; i < size; ++i) {
        if (blurred_cpu[i] != blurred_gpu[i]) {
            printf("Error: Mismatch at index %u, CPU: %u, GPU: %u\n", i, blurred_cpu[i], blurred_gpu[i]);
            break;
        }
    }

    // Free memory
    free(image);
    free(blurred_cpu);
    free(blurred_gpu);
    return 0;
}