#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 128

struct CSRGraph{
    unsigned int numVertices;
    unsigned int numEdges;
    unsigned int* srcPtrs;
    unsigned int* dst;
};

void bfs_cpu(CSRGraph csrGraph, unsigned int* level, unsigned int* newVertexVisited, unsigned int currLevel) {
    *newVertexVisited = 0;
    for (unsigned int vertex = 0; vertex < csrGraph.numVertices; ++vertex) {
        if (level[vertex] == UINT_MAX) {
            for (unsigned int i = csrGraph.srcPtrs[vertex]; i < csrGraph.srcPtrs[vertex + 1]; ++i) {
                unsigned int neighbor = csrGraph.dst[i];
                if (level[neighbor] == currLevel - 1) {
                    level[vertex] = currLevel;
                    *newVertexVisited = 1;
                    break;
                }
            }
        }
    }
}


// sparse matrix-vector multiplication
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level, unsigned int* newVertexVisted, unsigned int currLevel){
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if(vertex < csrGraph.numVertices){
        if(level[vertex] == UINT_MAX){
            for(unsigned int i = csrGraph.srcPtrs[vertex]; i < csrGraph.srcPtrs[vertex + 1]; ++i){
                unsigned int neighbor = csrGraph.dst[i];
                if(level[neighbor] == currLevel - 1){
                    level[vertex] = currLevel;
                    *newVertexVisted = 1;
                    break;
                }
            }
        }
    }
}

void bfs_gpu(CSRGraph csrGraph, unsigned int srcVertex, unsigned int* level){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    CSRGraph csrGraph_d;
    csrGraph_d.numVertices = csrGraph.numVertices;
    csrGraph_d.numEdges = csrGraph.numEdges;
    cudaMalloc((void**) &csrGraph_d.srcPtrs, (csrGraph_d.numVertices + 1) * sizeof(unsigned int));
    cudaMalloc((void**) &csrGraph_d.dst, csrGraph_d.numEdges * sizeof(unsigned int));
    unsigned int* level_d;
    cudaMalloc((void**) &level_d, csrGraph_d.numVertices * sizeof(unsigned int));
    unsigned int* newVertexVisted_d;
    cudaMalloc((void**) &newVertexVisted_d, sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(csrGraph_d.srcPtrs, csrGraph.srcPtrs,(csrGraph_d.numVertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrGraph_d.dst, csrGraph.dst,csrGraph_d.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    level[srcVertex] = 0;
    cudaMemcpy(level_d,level, csrGraph_d.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // calll kernel
    startTime(&timer);
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = (csrGraph_d.numVertices + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned int newVertexVisited = 1;
    for(unsigned int currLevel = 1; newVertexVisited; ++currLevel){
        newVertexVisited = 0;
        cudaMemcpy(newVertexVisted_d, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
        bfs_kernel<<<numBlocks, numThreadsPerBlock>>>(csrGraph_d, level_d, newVertexVisted_d, currLevel);
        cudaMemcpy(&newVertexVisited, newVertexVisted_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time",GREEN);

    // Copy data from GPU memory
    startTime(&timer);
    cudaMemcpy(level, level_d, csrGraph_d.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(csrGraph_d.srcPtrs);
    cudaFree(csrGraph_d.dst);
    cudaFree(level_d);
    cudaFree(newVertexVisted_d);
}


void initializeGraph(CSRGraph* csrGraph, unsigned int numVertices, unsigned int numEdges) {
    // Initialize random seed
    srand(time(NULL));

    // Allocate memory for adjacency matrix
    unsigned int** adjMatrix = (unsigned int**)malloc(numVertices * sizeof(unsigned int*));
    for (unsigned int i = 0; i < numVertices; ++i) {
        adjMatrix[i] = (unsigned int*)calloc(numVertices, sizeof(unsigned int));
    }

    // Randomly initialize the adjacency matrix with numEdges edges
    unsigned int edgeCount = 0;
    while (edgeCount < numEdges) {
        unsigned int src = rand() % numVertices;
        unsigned int dst = rand() % numVertices;
        if (src != dst && adjMatrix[src][dst] == 0) {
            adjMatrix[src][dst] = 1;
            edgeCount++;
        }
    }

    // Allocate memory for CSR format
    csrGraph->numVertices = numVertices;
    csrGraph->numEdges = numEdges;
    csrGraph->srcPtrs = (unsigned int*)malloc((numVertices + 1) * sizeof(unsigned int));
    csrGraph->dst = (unsigned int*)malloc(numEdges * sizeof(unsigned int));

    // Convert adjacency matrix to CSR format
    unsigned int edgeIndex = 0;
    csrGraph->srcPtrs[0] = 0;
    for (unsigned int i = 0; i < numVertices; ++i) {
        for (unsigned int j = 0; j < numVertices; ++j) {
            if (adjMatrix[i][j] == 1) {
                csrGraph->dst[edgeIndex++] = j;
            }
        }
        csrGraph->srcPtrs[i + 1] = edgeIndex;
    }

    // Free adjacency matrix memory
    for (unsigned int i = 0; i < numVertices; ++i) {
        free(adjMatrix[i]);
    }
    free(adjMatrix);
}

int main(int argc, char** argv) {
    Timer timer;

    // Initialize Graph and convert to CSR format
    unsigned int numVertices = 10000;
    unsigned int numEdges = 100000;
    CSRGraph csrGraph;
    initializeGraph(&csrGraph, numVertices, numEdges);

    // Allocate memory for level arrays
    unsigned int* level_cpu = (unsigned int*)malloc(numVertices * sizeof(unsigned int));
    unsigned int* level_gpu = (unsigned int*)malloc(numVertices * sizeof(unsigned int));
    for (unsigned int i = 0; i < numVertices; ++i) {
        level_cpu[i] = UINT_MAX;
        level_gpu[i] = UINT_MAX;
    }
    unsigned int srcVertex = 0;
    level_cpu[srcVertex] = 0;
    level_gpu[srcVertex] = 0;

    // CPU version
    startTime(&timer);
    unsigned int newVertexVisited = 1;
    for (unsigned int currLevel = 1; newVertexVisited; ++currLevel) {
        bfs_cpu(csrGraph, level_cpu, &newVertexVisited, currLevel);
    }
    stopTime(&timer);
    printElapsedTime(timer, "CPU BFS", GREEN);

    // GPU version
    startTime(&timer);
    bfs_gpu(csrGraph, srcVertex, level_gpu);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU BFS", GREEN);

    // Verify results
    float maxError = 0.0f;
    for (unsigned int i = 0; i < numVertices; ++i) {
        float error = fabs(level_cpu[i] - level_gpu[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    printf("Max error: %f\n", maxError);

    // Free memory
    free(csrGraph.srcPtrs);
    free(csrGraph.dst);
    free(level_cpu);
    free(level_gpu);

    return 0;
}