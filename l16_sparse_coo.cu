#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 128

struct COOMatrix{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* rowIdxs;
    unsigned int* colIdxs;
    float* values;
};

void spmv_coo_cpu(COOMatrix cooMatrix, float* inVector, float* outVector) {
    for (unsigned int i = 0; i < cooMatrix.numNonzeros; ++i) {
        unsigned int row = cooMatrix.rowIdxs[i];
        unsigned int col = cooMatrix.colIdxs[i];
        float value = cooMatrix.values[i];
        outVector[row] += inVector[col] * value;
    }
}


// sparse matrix-vector multiplication
__global__ void spmv_coo_kenel(COOMatrix cooMatrix, float* inVector, float* outVector){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < cooMatrix.numNonzeros){
        unsigned int row = cooMatrix.rowIdxs[i];
        unsigned int col = cooMatrix.colIdxs[i];
        float value = cooMatrix.values[i];
        atomicAdd(&outVector[row], inVector[col]*value);
    }
}

void spmv_coo_gpu(COOMatrix cooMatrix, float* inVector, float* outVector){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    COOMatrix cooMatrix_d;
    cooMatrix_d.numRows = cooMatrix.numRows;
    cooMatrix_d.numCols = cooMatrix.numCols;
    cooMatrix_d.numNonzeros = cooMatrix.numNonzeros;
    cudaMalloc((void**) &cooMatrix_d.rowIdxs, cooMatrix_d.numNonzeros*sizeof(unsigned int));
    cudaMalloc((void**) &cooMatrix_d.colIdxs, cooMatrix_d.numNonzeros*sizeof(unsigned int));
    cudaMalloc((void**) &cooMatrix_d.values, cooMatrix_d.numNonzeros*sizeof(float));
    float* inVector_d;
    cudaMalloc((void**) &inVector_d, cooMatrix_d.numCols*sizeof(float));
    float* outVector_d;
    cudaMalloc((void**) &outVector_d, cooMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(cooMatrix_d.rowIdxs, cooMatrix.rowIdxs, cooMatrix_d.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooMatrix_d.colIdxs, cooMatrix.colIdxs, cooMatrix_d.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooMatrix_d.values, cooMatrix.values, cooMatrix_d.numNonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inVector_d, inVector, cooMatrix_d.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(outVector_d, 0, cooMatrix_d.numRows * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // calll kernel
    startTime(&timer);
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = (cooMatrix_d.numNonzeros + BLOCK_DIM - 1)/BLOCK_DIM;
    spmv_coo_kenel<<<numBlocks, numThreadsPerBlock>>>(cooMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(outVector, outVector_d, cooMatrix_d.numRows*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(cooMatrix_d.rowIdxs);
    cudaFree(cooMatrix_d.colIdxs);
    cudaFree(cooMatrix_d.values);
    cudaFree(inVector_d);
    cudaFree(outVector_d);
}

void initializeSparseMatrix(COOMatrix* cooMatrix, unsigned int numRows, unsigned int numCols, unsigned int numNonzeros) {
    cooMatrix->numRows = numRows;
    cooMatrix->numCols = numCols;
    cooMatrix->numNonzeros = numNonzeros;

    cooMatrix->rowIdxs = (unsigned int*) malloc(numNonzeros * sizeof(unsigned int));
    cooMatrix->colIdxs = (unsigned int*) malloc(numNonzeros * sizeof(unsigned int));
    cooMatrix->values = (float*) malloc(numNonzeros * sizeof(float));

    // Initialize with some random values (for example purposes)
    for (unsigned int i = 0; i < numNonzeros; ++i) {
        cooMatrix->rowIdxs[i] = rand() % numRows;
        cooMatrix->colIdxs[i] = rand() % numCols;
        cooMatrix->values[i] = (float) rand() / RAND_MAX;
    }
}

int main(int argc, char** argv) {
    Timer timer;

    // Initialize sparse matrix and convert to COO format
    unsigned int numRows = 10000;
    unsigned int numCols = 10000;
    unsigned int numNonzeros = 100000;
    COOMatrix cooMatrix;
    initializeSparseMatrix(&cooMatrix, numRows, numCols, numNonzeros);

    // Initialize input vector
    float* inVector = (float*) malloc(numCols * sizeof(float));
    for (unsigned int i = 0; i < numCols; ++i) {
        inVector[i] = (float) rand() / RAND_MAX;
    }

    // Initialize output vectors
    float* outVector_cpu = (float*) calloc(numRows, sizeof(float));
    float* outVector_gpu = (float*) calloc(numRows, sizeof(float));

    // CPU version
    startTime(&timer);
    spmv_coo_cpu(cooMatrix, inVector, outVector_cpu);
    stopTime(&timer);
    printElapsedTime(timer, "CPU merge", GREEN);

    // GPU version
    startTime(&timer);
    spmv_coo_gpu(cooMatrix, inVector, outVector_gpu);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU merge", GREEN);

    // Verify results
    float maxError = 0.0f;
    for (unsigned int i = 0; i < numRows; ++i) {
        float error = fabs(outVector_cpu[i] - outVector_gpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    printf("Max error: %f\n", maxError);

    // Free memory
    free(cooMatrix.rowIdxs);
    free(cooMatrix.colIdxs);
    free(cooMatrix.values);
    free(inVector);
    free(outVector_cpu);
    free(outVector_gpu);
}