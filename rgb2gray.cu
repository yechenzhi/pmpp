#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

// CPU version of RGB to grayscale conversion
void rgb2gray_cpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    for (unsigned int i = 0; i < height; ++i) {
        for (unsigned int j = 0; j < width; ++j) {
            unsigned int idx = i * width + j;
            gray[idx] = red[idx] * 3 / 10 + green[idx] * 6 / 10 + blue[idx] * 1 / 10;
        }
    }
}

__global__ void rbg2gray_kenel(unsigned char* red, unsigned char* green, unsigned char* blue,  unsigned char* gray, unsigned int width, unsigned int height){
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row<height && col<width){
        unsigned int i = row*width+col;
        gray[i] = red[i]*3/10+green[i]*6/10+blue[i]*1/10;
    }
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue,  unsigned char* gray, unsigned int width, unsigned int height){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**) &red_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &green_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &blue_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &gray_d, width*height*sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(red_d, red, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // Perform computation on GPU
    startTime(&timer);
    // const unsigned int numThresaPerBlock = 512;
    // const unsigned int numBlocks = (N + numThresaPerBlock - 1) / numThresaPerBlock;
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    rbg2gray_kenel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, gray_d, gray_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",BLUE);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(gray, gray_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
}

int main(int argc, char** argv) {
    unsigned int width = 1 << 15;
    unsigned int height = 1 << 15;
    unsigned int size = width * height;

    // Allocate memory for RGB and gray images
    unsigned char* red = (unsigned char*) malloc(size * sizeof(unsigned char));
    unsigned char* green = (unsigned char*) malloc(size * sizeof(unsigned char));
    unsigned char* blue = (unsigned char*) malloc(size * sizeof(unsigned char));
    unsigned char* gray_cpu = (unsigned char*) malloc(size * sizeof(unsigned char));
    unsigned char* gray_gpu = (unsigned char*) malloc(size * sizeof(unsigned char));

    // Initialize RGB arrays with some values
    for (unsigned int i = 0; i < size; ++i) {
        red[i] = rand() % 256;
        green[i] = rand() % 256;
        blue[i] = rand() % 256;
    }

    Timer timer;

    // CPU version
    startTime(&timer);
    rgb2gray_cpu(red, green, blue, gray_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU RGB to Gray", GREEN);

    // GPU version
    startTime(&timer);
    rgb2gray_gpu(red, green, blue, gray_gpu, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU RGB to Gray", BLUE);

    // Verify results
    for (unsigned int i = 0; i < size; ++i) {
        if (gray_cpu[i] != gray_gpu[i]) {
            printf("Error: Mismatch at index %u, CPU: %u, GPU: %u\n", i, gray_cpu[i], gray_gpu[i]);
            break;
        }
    }

    // Free memory
    free(red);
    free(green);
    free(blue);
    free(gray_cpu);
    free(gray_gpu);

    return 0;
}