#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 128

struct ELLMatrix{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int maxNNZPerRow; // max number of nonzeros per row
    unsigned int* nnzPerRow; // number of nonzeros per row
    unsigned int* colIdxs; // column major order
    float* values; // column major order
};

void spmv_ell_cpu(ELLMatrix ellMatrix, float* inVector, float* outVector) {
    for (unsigned int row = 0; row < ellMatrix.numRows; ++row) {
        float sum = 0.0f;
        for (unsigned int nnzIdx = 0; nnzIdx < ellMatrix.nnzPerRow[row]; ++nnzIdx) {
            unsigned int i = nnzIdx * ellMatrix.numRows + row;
            float value = ellMatrix.values[i];
            unsigned int col = ellMatrix.colIdxs[i];
            sum += value * inVector[col];
        }
        outVector[row] = sum;
    }
}


// sparse matrix-vector multiplication
__global__ void spmv_ell_kenel(ELLMatrix ellMatrix, float* inVector, float* outVector){
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<ellMatrix.numRows){
        float sum = 0.0f;
        for(unsigned int nnzIdx = 0; nnzIdx < ellMatrix.nnzPerRow[row]; ++nnzIdx){
            unsigned int i = nnzIdx * ellMatrix.numRows + row;
            float value = ellMatrix.values[i];
            unsigned int col = ellMatrix.colIdxs[i];
            sum += value * inVector[col];
        }
        outVector[row] = sum;
    }
}

void spmv_ell_gpu(ELLMatrix ellMatrix, float* inVector, float* outVector){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    ELLMatrix ellMatrix_d;
    ellMatrix_d.numRows = ellMatrix.numRows;
    ellMatrix_d.numCols = ellMatrix.numCols;
    ellMatrix_d.maxNNZPerRow = ellMatrix.maxNNZPerRow;
    cudaMalloc((void**) &ellMatrix_d.nnzPerRow, ellMatrix_d.numRows * sizeof(unsigned int));
    cudaMalloc((void**) &ellMatrix_d.colIdxs, ellMatrix_d.numRows * ellMatrix_d.maxNNZPerRow*sizeof(unsigned int));
    cudaMalloc((void**) &ellMatrix_d.values, ellMatrix_d.numRows * ellMatrix_d.maxNNZPerRow*sizeof(float));
    float* inVector_d;
    cudaMalloc((void**) &inVector_d, ellMatrix_d.numCols*sizeof(float));
    float* outVector_d;
    cudaMalloc((void**) &outVector_d, ellMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(ellMatrix_d.nnzPerRow, ellMatrix.nnzPerRow, ellMatrix_d.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(ellMatrix_d.colIdxs, ellMatrix.colIdxs, ellMatrix_d.numRows * ellMatrix_d.maxNNZPerRow * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(ellMatrix_d.values, ellMatrix.values, ellMatrix_d.numRows * ellMatrix_d.maxNNZPerRow * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inVector_d, inVector, ellMatrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(outVector_d, 0, ellMatrix.numRows * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // calll kernel
    startTime(&timer);
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = (ellMatrix_d.numRows * ellMatrix_d.maxNNZPerRow + BLOCK_DIM - 1) / BLOCK_DIM;
    spmv_ell_kenel<<<numBlocks, numThreadsPerBlock>>>(ellMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(outVector, outVector_d, ellMatrix_d.numRows*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(ellMatrix_d.nnzPerRow);
    cudaFree(ellMatrix_d.colIdxs);
    cudaFree(ellMatrix_d.values);
    cudaFree(inVector_d);
    cudaFree(outVector_d);
}

void initializeSparseMatrix(ELLMatrix* ellMatrix, unsigned int numRows, unsigned int numCols, unsigned int numNonzeros) {
    ellMatrix->numRows = numRows;
    ellMatrix->numCols = numCols;
    ellMatrix->maxNNZPerRow = 0;

    // Initialize nnzPerRow with zeros
    ellMatrix->nnzPerRow = (unsigned int*) calloc(numRows, sizeof(unsigned int));

    // Temporary arrays to store non-zero elements
    unsigned int** tempColIdxs = (unsigned int**) malloc(numRows * sizeof(unsigned int*));
    float** tempValues = (float**) malloc(numRows * sizeof(float*));
    for (unsigned int i = 0; i < numRows; ++i) {
        tempColIdxs[i] = (unsigned int*) malloc(numCols * sizeof(unsigned int));
        tempValues[i] = (float*) malloc(numCols * sizeof(float));
    }

    // Initialize with some random values (for example purposes)
    for (unsigned int i = 0; i < numNonzeros; ++i) {
        unsigned int row = rand() % numRows;
        unsigned int col = rand() % numCols;
        float value = (float) rand() / RAND_MAX;

        // Update nnzPerRow and maxNNZPerRow
        if (ellMatrix->nnzPerRow[row] < numCols) {
            tempColIdxs[row][ellMatrix->nnzPerRow[row]] = col;
            tempValues[row][ellMatrix->nnzPerRow[row]] = value;
            ellMatrix->nnzPerRow[row]++;
            if (ellMatrix->nnzPerRow[row] > ellMatrix->maxNNZPerRow) {
                ellMatrix->maxNNZPerRow = ellMatrix->nnzPerRow[row];
            }
        }
    }

    // Allocate colIdxs and values arrays
    ellMatrix->colIdxs = (unsigned int*) malloc(numRows * ellMatrix->maxNNZPerRow * sizeof(unsigned int));
    ellMatrix->values = (float*) malloc(numRows * ellMatrix->maxNNZPerRow * sizeof(float));

    // Initialize colIdxs and values arrays in column major order
    for (unsigned int col = 0; col < ellMatrix->maxNNZPerRow; ++col) {
        for (unsigned int row = 0; row < numRows; ++row) {
            unsigned int index = col * numRows + row;
            if (col < ellMatrix->nnzPerRow[row]) {
                ellMatrix->colIdxs[index] = tempColIdxs[row][col];
                ellMatrix->values[index] = tempValues[row][col];
            } else {
                ellMatrix->colIdxs[index] = 0;
                ellMatrix->values[index] = 0.0f;
            }
        }
    }
}

int main(int argc, char** argv) {
    Timer timer;

    // Initialize sparse matrix and convert to COO format
    unsigned int numRows = 10000;
    unsigned int numCols = 10000;
    unsigned int numNonzeros = 100000;
    ELLMatrix ellMatrix;
    initializeSparseMatrix(&ellMatrix, numRows, numCols, numNonzeros);

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
    spmv_ell_cpu(ellMatrix, inVector, outVector_cpu);
    stopTime(&timer);
    printElapsedTime(timer, "CPU merge", GREEN);

    // GPU version
    startTime(&timer);
    spmv_ell_gpu(ellMatrix, inVector, outVector_gpu);
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
    free(ellMatrix.nnzPerRow);
    free(ellMatrix.colIdxs);
    free(ellMatrix.values);
    free(inVector);
    free(outVector_cpu);
    free(outVector_gpu);
}