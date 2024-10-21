#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 128

struct COOGraph{
    unsigned int numVertices;
    unsigned int numEdges;
    unsigned int* src;
    unsigned int* dst;
};

void bfs_cpu(COOGraph cooGraph, unsigned int* level, unsigned int* newVertexVisited, unsigned int currLevel) {
    *newVertexVisited = 0;
    for (unsigned int edge = 0; edge < cooGraph.numEdges; ++edge) {
        unsigned int src = cooGraph.src[edge];
        unsigned int dst = cooGraph.dst[edge];
        if (level[src] == currLevel - 1 && level[dst] == UINT_MAX) {
            level[dst] = currLevel;
            *newVertexVisited = 1;
        }
    }
}


__global__ void bfs_kernel(COOGraph cooGraph, unsigned int* level, unsigned int* newVertexVisited, unsigned int currLevel){
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if(edge < cooGraph.numEdges){
        unsigned int src = cooGraph.src[edge];
        unsigned int dst = cooGraph.dst[edge];
        if(level[src] == currLevel - 1 && level[dst] == UINT_MAX){
            level[dst] = currLevel;
            *newVertexVisited = 1;
        }
    }
}

void bfs_gpu(COOGraph cooGraph, unsigned int srcVertex, unsigned int* level){
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    COOGraph cooGraph_d;
    cooGraph_d.numVertices = cooGraph.numVertices;
    cooGraph_d.numEdges = cooGraph.numEdges;
    cudaMalloc((void**) &cooGraph_d.src, cooGraph_d.numEdges * sizeof(unsigned int));
    cudaMalloc((void**) &cooGraph_d.dst, cooGraph_d.numEdges * sizeof(unsigned int));
    unsigned int* level_d;
    cudaMalloc((void**) &level_d, cooGraph_d.numVertices * sizeof(unsigned int));
    unsigned int* newVertexVisted_d;
    cudaMalloc((void**) &newVertexVisted_d, sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time",BLUE);

    // Copy data to GPU memory
    startTime(&timer);
    cudaMemcpy(cooGraph_d.src, cooGraph.src,cooGraph_d.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooGraph_d.dst, cooGraph.dst,cooGraph_d.numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    level[srcVertex] = 0;
    cudaMemcpy(level_d,level, cooGraph_d.numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time",BLUE);

    // calll kernel
    startTime(&timer);
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = (cooGraph_d.numEdges + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned int newVertexVisited = 1;
    for(unsigned int currLevel = 1; newVertexVisited; ++currLevel){
        newVertexVisited = 0;
        cudaMemcpy(newVertexVisted_d, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
        bfs_kernel<<<numBlocks, numThreadsPerBlock>>>(cooGraph_d, level_d, newVertexVisted_d, currLevel);
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
    cudaMemcpy(level, level_d, cooGraph_d.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to CPU time",BLUE);

    // Deallocate GPU memory
    cudaFree(cooGraph_d.src);
    cudaFree(cooGraph_d.dst);
    cudaFree(level_d);
    cudaFree(newVertexVisted_d);
}


void initializeGraph(COOGraph* cooGraph, unsigned int numVertices, unsigned int numEdges) {
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

    // Allocate memory for COO format
    cooGraph->numVertices = numVertices;
    cooGraph->numEdges = numEdges;
    cooGraph->src = (unsigned int*)malloc(numEdges * sizeof(unsigned int));
    cooGraph->dst = (unsigned int*)malloc(numEdges * sizeof(unsigned int));

    // Convert adjacency matrix to COO format
    unsigned int edgeIndex = 0;
    cooGraph->src[0] = 0;
    for (unsigned int i = 0; i < numVertices; ++i) {
        for (unsigned int j = 0; j < numVertices; ++j) {
            if (adjMatrix[i][j] == 1) {
                cooGraph->src[edgeIndex] = i;
                cooGraph->dst[edgeIndex++] = j;
            }
        }
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
    COOGraph cooGraph;
    initializeGraph(&cooGraph, numVertices, numEdges);

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
        bfs_cpu(cooGraph, level_cpu, &newVertexVisited, currLevel);
    }
    stopTime(&timer);
    printElapsedTime(timer, "CPU BFS", GREEN);

    // GPU version
    startTime(&timer);
    bfs_gpu(cooGraph, srcVertex, level_gpu);
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
    free(cooGraph.src);
    free(cooGraph.dst);
    free(level_cpu);
    free(level_gpu);

    return 0;
}