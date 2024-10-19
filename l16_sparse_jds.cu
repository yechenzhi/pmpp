#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

#define BLOCK_DIM 128

struct JDSMatrix {
    unsigned int numRows;        // 矩阵的行数
    unsigned int numCols;        // 矩阵的列数
    unsigned int numNonzeros;    // 非零元素的总数
    unsigned int maxNNZPerRow;   // 每行最大非零元素数

    unsigned int* nnzPerRow;     // 每行非零元素的数量 [numRows]
    unsigned int* rowPerm;       // 行的重排序 [numRows]
    unsigned int* diagPtr;       // 每个对角线的起始位置 [maxNNZPerRow + 1]
    unsigned int* colIdxs;       // 列索引 [numNonzeros]
    float* values;      // 非零元素值 [numNonzeros]
};

void spmv_jds_cpu(JDSMatrix jdsMatrix, float* inVector, float* outVector) {
    for (unsigned int row = 0; row < jdsMatrix.numRows; ++row) {
        float sum = 0.0f;
        for (unsigned int iter = 0; iter < jdsMatrix.nnzPerRow[row]; ++iter) {
            unsigned int i = jdsMatrix.diagPtr[iter] + row;
            unsigned int col = jdsMatrix.colIdxs[i];
            float value = jdsMatrix.values[i];
            sum += value * inVector[col];
        }
        outVector[jdsMatrix.rowPerm[row]] = sum;
    }
}

// sparse matrix-vector multiplication
// __global__ void spmv_jds_kenel(JDSMatrix jdsMatrix, float* inVector, float* outVector){
//     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid < jdsMatrix.numRows){
//         float sum = 0.0f;
//         unsigned int row = jdsMatrix.rowPerm[tid];
//         for(unsigned int iter = 0; iter < jdsMatrix.nnzPerRow[row]; ++iter){
//             unsigned int i = jdsMatrix.diagPtr[iter] + tid;
//             unsigned int col = jdsMatrix.colIdxs[i];
//             float value = jdsMatrix.values[i];
//             sum += value * inVector[col];
//         }
//         outVector[row] = sum;
//     }
// }

__global__ void spmv_jds_kernel(JDSMatrix jdsMatrix, float* inVector, float* outVector){
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < jdsMatrix.numRows){
        float sum = 0.0f;
        for(unsigned int iter = 0; iter < jdsMatrix.nnzPerRow[row]; ++iter){
            unsigned int i = jdsMatrix.diagPtr[iter] + row;
            unsigned int col = jdsMatrix.colIdxs[i];
            float value = jdsMatrix.values[i];
            sum += value * inVector[col];
        }
        outVector[jdsMatrix.rowPerm[row]] = sum; // Corrected indexing
    }
}

void spmv_jds_gpu(JDSMatrix jdsMatrix, float* inVector, float* outVector){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    JDSMatrix jdsMatrix_d;
    jdsMatrix_d.numRows = jdsMatrix.numRows;
    jdsMatrix_d.numCols = jdsMatrix.numCols;
    jdsMatrix_d.numNonzeros = jdsMatrix.numNonzeros;
    jdsMatrix_d.maxNNZPerRow = jdsMatrix.maxNNZPerRow;
    cudaMalloc((void**) &jdsMatrix_d.nnzPerRow, jdsMatrix_d.numRows * sizeof(unsigned int));
    cudaMalloc((void**) &jdsMatrix_d.rowPerm, jdsMatrix_d.numRows * sizeof(unsigned int));
    cudaMalloc((void**) &jdsMatrix_d.diagPtr, (jdsMatrix_d.maxNNZPerRow + 1) * sizeof(unsigned int));
    cudaMalloc((void**) &jdsMatrix_d.colIdxs, jdsMatrix_d.numNonzeros * sizeof(unsigned int));
    cudaMalloc((void**) &jdsMatrix_d.values, jdsMatrix_d.numNonzeros * sizeof(float));

    float* inVector_d;
    cudaMalloc((void**) &inVector_d, jdsMatrix_d.numCols*sizeof(float));
    float* outVector_d;
    cudaMalloc((void**) &outVector_d, jdsMatrix_d.numRows*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(jdsMatrix_d.nnzPerRow, jdsMatrix.nnzPerRow, jdsMatrix_d.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(jdsMatrix_d.rowPerm, jdsMatrix.rowPerm, jdsMatrix_d.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(jdsMatrix_d.diagPtr, jdsMatrix.diagPtr, (jdsMatrix_d.maxNNZPerRow + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(jdsMatrix_d.colIdxs, jdsMatrix.colIdxs, jdsMatrix_d.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(jdsMatrix_d.values, jdsMatrix.values, jdsMatrix_d.numNonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inVector_d, inVector, jdsMatrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(outVector_d, 0, jdsMatrix.numRows * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // call kernel
    startTime(&timer);
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = (jdsMatrix_d.numRows + BLOCK_DIM - 1) / BLOCK_DIM;
    spmv_jds_kernel<<<numBlocks, numThreadsPerBlock>>>(jdsMatrix_d, inVector_d, outVector_d);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(outVector, outVector_d, jdsMatrix_d.numRows*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(jdsMatrix_d.nnzPerRow);
    cudaFree(jdsMatrix_d.rowPerm);
    cudaFree(jdsMatrix_d.diagPtr);
    cudaFree(jdsMatrix_d.colIdxs);
    cudaFree(jdsMatrix_d.values);
    cudaFree(inVector_d);
    cudaFree(outVector_d);
}

struct RowInfo {
    unsigned int rowIndex;
    unsigned int nnz;
};

int compareRowInfo(const void* a, const void* b) {
    RowInfo* ra = (RowInfo*)a;
    RowInfo* rb = (RowInfo*)b;
    if (ra->nnz < rb->nnz) return 1; // For decreasing order
    else if (ra->nnz > rb->nnz) return -1;
    else return 0;
}

void initializeSparseMatrix(JDSMatrix* jdsMatrix, unsigned int numRows, unsigned int numCols, unsigned int numNonzeros) {
    jdsMatrix->numRows = numRows;
    jdsMatrix->numCols = numCols;
    jdsMatrix->numNonzeros = numNonzeros;

    // Initialize nnzPerRow
    unsigned int* nnzPerRow = (unsigned int*)malloc(numRows * sizeof(unsigned int));
    for (unsigned int i = 0; i < numRows; ++i) {
        nnzPerRow[i] = 1;
    }
    unsigned int totalNNZAssigned = numRows;
    unsigned int numRemainingNNZ = numNonzeros - totalNNZAssigned;

    for (unsigned int k = 0; k < numRemainingNNZ; ++k) {
        unsigned int i = rand() % numRows;
        nnzPerRow[i] += 1;
    }

    // Compute maxNNZPerRow
    unsigned int maxNNZPerRow = 0;
    for (unsigned int i = 0; i < numRows; ++i) {
        if (nnzPerRow[i] > maxNNZPerRow) {
            maxNNZPerRow = nnzPerRow[i];
        }
    }
    jdsMatrix->maxNNZPerRow = maxNNZPerRow;

    // Create rowPerm array
    RowInfo* rowInfoArray = (RowInfo*)malloc(numRows * sizeof(RowInfo));
    for (unsigned int i = 0; i < numRows; ++i) {
        rowInfoArray[i].rowIndex = i;
        rowInfoArray[i].nnz = nnzPerRow[i];
    }
    qsort(rowInfoArray, numRows, sizeof(RowInfo), compareRowInfo);

    // Create rowPerm and rearranged nnzPerRow
    unsigned int* rowPerm = (unsigned int*)malloc(numRows * sizeof(unsigned int));
    for (unsigned int i = 0; i < numRows; ++i) {
        rowPerm[i] = rowInfoArray[i].rowIndex;
        nnzPerRow[i] = rowInfoArray[i].nnz; // Reuse nnzPerRow to store rearranged nnz
    }

    // Now, generate column indices and values
    unsigned int** rowColIdxs = (unsigned int**)malloc(numRows * sizeof(unsigned int*));
    float** rowValues = (float**)malloc(numRows * sizeof(float*));

    for (unsigned int i = 0; i < numRows; ++i) {
        unsigned int nnz = nnzPerRow[i];
        rowColIdxs[i] = (unsigned int*)malloc(nnz * sizeof(unsigned int));
        rowValues[i] = (float*)malloc(nnz * sizeof(float));
        unsigned int count = 0;
        while (count < nnz) {
            unsigned int col = rand() % numCols;
            // Check for duplicates
            int duplicate = 0;
            for (unsigned int j = 0; j < count; ++j) {
                if (rowColIdxs[i][j] == col) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate) {
                rowColIdxs[i][count] = col;
                rowValues[i][count] = (float)rand() / RAND_MAX;
                count++;
            }
        }
    }

    // Build diagLengths and diagPtr
    unsigned int* diagLengths = (unsigned int*)malloc(maxNNZPerRow * sizeof(unsigned int));
    for (unsigned int iter = 0; iter < maxNNZPerRow; ++iter) {
        diagLengths[iter] = 0;
    }
    for (unsigned int i = 0; i < numRows; ++i) {
        unsigned int nnz = nnzPerRow[i];
        for (unsigned int iter = 0; iter < nnz; ++iter) {
            diagLengths[iter] += 1;
        }
    }
    unsigned int* diagPtr = (unsigned int*)malloc((maxNNZPerRow + 1) * sizeof(unsigned int));
    diagPtr[0] = 0;
    for (unsigned int iter = 0; iter < maxNNZPerRow; ++iter) {
        diagPtr[iter + 1] = diagPtr[iter] + diagLengths[iter];
    }

    // Allocate colIdxs and values
    unsigned int* colIdxs = (unsigned int*)malloc(numNonzeros * sizeof(unsigned int));
    float* values = (float*)malloc(numNonzeros * sizeof(float));

    // Initialize diagCounts
    unsigned int* diagCounts = (unsigned int*)malloc(maxNNZPerRow * sizeof(unsigned int));
    for (unsigned int iter = 0; iter < maxNNZPerRow; ++iter) {
        diagCounts[iter] = 0;
    }

    // Fill colIdxs and values in JDS format
    for (unsigned int i = 0; i < numRows; ++i) {
        unsigned int nnz = nnzPerRow[i];
        for (unsigned int iter = 0; iter < nnz; ++iter) {
            unsigned int pos = diagPtr[iter] + diagCounts[iter];
            colIdxs[pos] = rowColIdxs[i][iter];
            values[pos] = rowValues[i][iter];
            diagCounts[iter] += 1;
        }
    }

    // Assign to jdsMatrix
    jdsMatrix->nnzPerRow = nnzPerRow;
    jdsMatrix->rowPerm = rowPerm;
    jdsMatrix->diagPtr = diagPtr;
    jdsMatrix->colIdxs = colIdxs;
    jdsMatrix->values = values;
}

int main(int argc, char** argv) {
    Timer timer;

    // Initialize sparse matrix and convert to JDS format
    unsigned int numRows = 10000;
    unsigned int numCols = 10000;
    unsigned int numNonzeros = 100000;
    JDSMatrix jdsMatrix;
    initializeSparseMatrix(&jdsMatrix, numRows, numCols, numNonzeros);

    printf("Matrix initialized. numRows: %u, numCols: %u, numNonzeros: %u, maxNNZPerRow: %u\n",
           jdsMatrix.numRows, jdsMatrix.numCols, jdsMatrix.numNonzeros, jdsMatrix.maxNNZPerRow);
    
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
    spmv_jds_cpu(jdsMatrix, inVector, outVector_cpu);
    stopTime(&timer);
    printElapsedTime(timer, "CPU merge", GREEN);
    
    printf("Matrix initialized. numRows: %u, numCols: %u, numNonzeros: %u, maxNNZPerRow: %u\n",
           jdsMatrix.numRows, jdsMatrix.numCols, jdsMatrix.numNonzeros, jdsMatrix.maxNNZPerRow); 

    // GPU version
    startTime(&timer);
    spmv_jds_gpu(jdsMatrix, inVector, outVector_gpu);
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
    free(jdsMatrix.nnzPerRow);
    free(jdsMatrix.rowPerm);
    free(jdsMatrix.diagPtr);
    free(jdsMatrix.colIdxs);
    free(jdsMatrix.values);
    free(inVector);
    free(outVector_cpu);
    free(outVector_gpu);

    return 0;
}