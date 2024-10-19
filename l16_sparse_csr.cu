#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 128

struct CSRMatrix{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* rowPtrs;
    unsigned int* colIdxs;
    float* values;
};

void spmv_csr_cpu(CSRMatrix csrMatrix, float* inVector, float* outVector) {
    for (unsigned int row = 0; row < csrMatrix.numRows; ++row) {
        float sum = 0.0f;
        for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i) {
            unsigned int col = csrMatrix.colIdxs[i];
            float value = csrMatrix.values[i];
            sum += value * inVector[col];
        }
        outVector[row] = sum;
    }
}


// sparse matrix-vector multiplication
__global__ void spmv_csr_kenel(CSRMatrix csrMatrix, float* inVector, float* outVector){
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < csrMatrix.numRows){
        float sum = 0.0f;
        for(unsigned int i=csrMatrix.rowPtrs[row]; i<csrMatrix.rowPtrs[row+1];++i){
            unsigned int col = csrMatrix.colIdxs[i];
            float value = csrMatrix.values[i];
            sum += value * inVector[col];
        }
        outVector[row] = sum;
    }
}

void spmv_csr_gpu(CSRMatrix csrMatrix, float* inVector, float* outVector){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    CSRMatrix csrMatrix_d;
    csrMatrix_d.numRows = csrMatrix.numRows;
    csrMatrix_d.numCols = csrMatrix.numCols;
    csrMatrix_d.numNonzeros = csrMatrix.numNonzeros;
    cudaMalloc((void**) &csrMatrix_d.rowPtrs, (csrMatrix_d.numRows + 1) * sizeof(unsigned int));
    cudaMalloc((void**) &csrMatrix_d.colIdxs, csrMatrix_d.numNonzeros*sizeof(unsigned int));
    cudaMalloc((void**) &csrMatrix_d.values, csrMatrix_d.numNonzeros*sizeof(float));
    float* inVector_d;
    cudaMalloc((void**) &inVector_d, csrMatrix_d.numCols*sizeof(float));
    float* outVector_d;
    cudaMalloc((void**) &outVector_d, csrMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(csrMatrix_d.rowPtrs, csrMatrix_d.rowPtrs, (csrMatrix_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrMatrix_d.colIdxs, csrMatrix_d.colIdxs, csrMatrix_d.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrMatrix_d.values, csrMatrix_d.values, csrMatrix_d.numNonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inVector_d, inVector, csrMatrix_d.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(outVector_d, 0, csrMatrix_d.numRows * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // calll kernel
    startTime(&timer);
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = (csrMatrix_d.numRows + BLOCK_DIM - 1) / BLOCK_DIM;
    spmv_csr_kenel<<<numBlocks, numThreadsPerBlock>>>(csrMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(outVector, outVector_d, csrMatrix_d.numRows*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(csrMatrix_d.rowPtrs);
    cudaFree(csrMatrix_d.colIdxs);
    cudaFree(csrMatrix_d.values);
    cudaFree(inVector_d);
    cudaFree(outVector_d);
}

void initializeSparseMatrix(CSRMatrix* csrMatrix, unsigned int numRows, unsigned int numCols, unsigned int numNonzeros) {
    csrMatrix->numRows = numRows;
    csrMatrix->numCols = numCols;
    csrMatrix->numNonzeros = numNonzeros;

    csrMatrix->rowPtrs = (unsigned int*) malloc((numRows + 1) * sizeof(unsigned int));
    csrMatrix->colIdxs = (unsigned int*) malloc(numNonzeros * sizeof(unsigned int));
    csrMatrix->values = (float*) malloc(numNonzeros * sizeof(float));

    // Initialize rowPtrs with zeros
    for (unsigned int i = 0; i <= numRows; ++i) {
        csrMatrix->rowPtrs[i] = 0;
    }

    // Initialize with some random values (for example purposes)
    for (unsigned int i = 0; i < numNonzeros; ++i) {
        unsigned int row = rand() % numRows;
        unsigned int col = rand() % numCols;
        float value = (float) rand() / RAND_MAX;

        // Insert into CSR format
        csrMatrix->colIdxs[i] = col;
        csrMatrix->values[i] = value;
        csrMatrix->rowPtrs[row + 1]++;
    }

    // Accumulate rowPtrs
    for (unsigned int i = 1; i <= numRows; ++i) {
        csrMatrix->rowPtrs[i] += csrMatrix->rowPtrs[i - 1];
    }
}

int main(int argc, char** argv) {
    Timer timer;

    // Initialize sparse matrix and convert to COO format
    unsigned int numRows = 10000;
    unsigned int numCols = 10000;
    unsigned int numNonzeros = 100000;
    CSRMatrix CsrMatrix;
    initializeSparseMatrix(&CsrMatrix, numRows, numCols, numNonzeros);

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
    spmv_csr_cpu(CsrMatrix, inVector, outVector_cpu);
    stopTime(&timer);
    printElapsedTime(timer, "CPU merge", GREEN);

    // GPU version
    startTime(&timer);
    spmv_csr_gpu(CsrMatrix, inVector, outVector_gpu);
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
    free(CsrMatrix.rowPtrs);
    free(CsrMatrix.colIdxs);
    free(CsrMatrix.values);
    free(inVector);
    free(outVector_cpu);
    free(outVector_gpu);
}