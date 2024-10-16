#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 128
#define COARSE_FACTOR 6

__device__ __host__ void mergeSequential(float* A, float* B, float* C, unsigned int m, unsigned int n) {
    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int k = 0;
    while(i < m && j < n){
        if(A[i] <= B[j]){
            C[k++] = A[i++];
        }
        else{
            C[k++] = B[j++];
        }
    }
    while(i<m){
        C[k++] = A[i++];
    }
    while(j<n){
        C[k++] = B[j++];
    }
}

__device__ unsigned int coRank(float* A, float* B, unsigned int m, unsigned int n, unsigned int k){
    unsigned int iLow = k>n?k-n:0;
    unsigned int iHigh = k>m?m:k;
    while(true){
        unsigned int i = (iLow+iHigh) / 2;
        unsigned int j = k - i;
        if(i > 0 && j < n && A[i-1] > B[j]){
            iHigh = i - 1;
        }
        else if(j > 0 && i < m && B[j-1] >= A[i]){
            iLow = i + 1;
        } else {
            return i;
        }
    }
}

__global__ void merge_kenel(float* A, float* B, float* C, unsigned int m, unsigned int n){

    // Find block segments
    unsigned int kBlock = blockIdx.x*blockDim.x*COARSE_FACTOR;
    unsigned int kNextBlock = (blockIdx.x<gridDim.x-1)?(kBlock+blockDim.x*COARSE_FACTOR):(m+n);
    __shared__ unsigned int iBlock;
    __shared__ unsigned int iNextBlock;
    if(threadIdx.x == 0){
        iBlock = coRank(A,B,m,n,kBlock);
        iNextBlock =coRank(A,B,m,n,kNextBlock);
    }
    __syncthreads();
    unsigned int jBlock = kBlock - iBlock;
    unsigned int jNextBlock = kNextBlock - iNextBlock;

    // load block segemnts
    __shared__ float A_s[COARSE_FACTOR*BLOCK_DIM];
    unsigned int mBlock = iNextBlock - iBlock;
    for(unsigned int i = threadIdx.x; i < mBlock; i+= BLOCK_DIM){
        A_s[i] = A[iBlock+i];
    }
    float* B_s = A_s + mBlock;
    unsigned int nBlock = jNextBlock - jBlock;
    for(unsigned int j = threadIdx.x; j < nBlock; j+= BLOCK_DIM){
        B_s[j] = B[jBlock+j];
    }
    __syncthreads();

    // merge in shared memory 
    __shared__ float C_s[COARSE_FACTOR*BLOCK_DIM];
    unsigned int k = threadIdx.x*COARSE_FACTOR;
    if(k<mBlock+nBlock){
        unsigned int i = coRank(A_s,B_s,mBlock,nBlock,k);
        unsigned int j = k - i; 
        unsigned int kNext = (k+COARSE_FACTOR < mBlock+nBlock)?(k+COARSE_FACTOR):(mBlock+nBlock);
        unsigned int iNext = coRank(A_s,B_s,mBlock,nBlock,kNext);
        unsigned int jNext =kNext - iNext;
        mergeSequential(&A_s[i], &B_s[j], &C_s[k],iNext-i,jNext-j);
    }
    __syncthreads();

    // write back to global memory 
    for(unsigned int k = threadIdx.x; k < mBlock+nBlock; k+= BLOCK_DIM){
        C[kBlock+k] = C_s[k];
    }
}

void merge_gpu(float* A, float* B, float* C, unsigned int m, unsigned int n){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *A_d;
    float *B_d;
    float *C_d;;
    cudaMalloc((void**) &A_d, m*sizeof(float));
    cudaMalloc((void**) &B_d, n*sizeof(float));
    cudaMalloc((void**) &C_d, (m+n)*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(A_d, A, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // calll kernel
    startTime(&timer);
    // const unsigned int numThresaPerBlock = 512;
    // const unsigned int numBlocks = (N + numThresaPerBlock - 1) / numThresaPerBlock;
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numElementsPerBlock = numThreadsPerBlock * COARSE_FACTOR;
    unsigned int numBlocks = ((m+n) + numElementsPerBlock - 1)/numElementsPerBlock;
    merge_kenel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, m, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(C, C_d, (m+n)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int compare(const void *a, const void *b) {
    return (*(float*)a > *(float*)b) - (*(float*)a < *(float*)b);
}

int main(int argc, char** argv) {
    unsigned int m = 1024*1024*100;
    unsigned int n = 1024*1024*100;

    float* A = (float*) malloc(m * sizeof(float));
    float* B = (float*) malloc(n * sizeof(float));
    float* C_cpu = (float*) malloc((m+n) * sizeof(float));
    float* C_gpu = (float*) malloc((m+n) * sizeof(float));

    for (unsigned int i = 0; i < m; ++i) {
        A[i] = (float)rand();
    }

    for (unsigned int i = 0; i < n; ++i) {
        B[i] = (float)rand();
    }

    qsort(A, m, sizeof(float), compare);
    qsort(B, n, sizeof(float), compare);
    Timer timer;

    // CPU version
    startTime(&timer);
    mergeSequential(A,B,C_cpu,m,n);
    stopTime(&timer);
    printElapsedTime(timer, "CPU merge", GREEN);

    // GPU version
    startTime(&timer);
    merge_gpu(A,B,C_gpu,m,n);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU merge", GREEN);

    // Verify results
    float maxError = 0.0f;
    for (unsigned int i = 0; i < m+n; ++i) {
        float error = fabs(C_cpu[i] - C_gpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    printf("Max error: %f\n", maxError);
    printf("Array A (sorted):\n");
    for (unsigned int i = 0; i < 10; i++) {
        printf("%f ", A[i]);
    }
    printf("\n");

    printf("Array B (sorted):\n");
    for (unsigned int i = 0; i < 10; i++) {
        printf("%f ", B[i]);
    }
    printf("\n");

    printf("Array C (sorted):\n");
    for (unsigned int i = 0; i < 10; i++) {
        printf("%f ", C_cpu[i]);
    }
    printf("\n");

    // Free memory
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    return 0;
}