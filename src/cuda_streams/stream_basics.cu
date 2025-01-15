#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); \
        exit(EXIT_FAILURE); \
    }

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define D2D cudaMemcpyDeviceToDevice
#define H2H cudaMemcpyHostToHost

// Learning about streams through a basic Async copying of memory
__global__ void vectorAddKernel(float *A, float *B, float *C, int n)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < n)
    {
        C[idx] = A[idx] +  B[idx];
    }
}

void vectorAdd(float *A, float *B, float *C, int n)
{
    float *d_A, *d_B, *d_C;
    size_t size = n*sizeof(float);
    cudaStream_t stream1, stream2;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, size));

    // Create streams
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    // Copy inputs to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, A, size, H2D, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, B, size, H2D, stream2));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1)/blockSize;   //ceil(n/blockSize)

    vectorAddKernel<<<gridSize, blockSize, 0, stream1>>>(d_A, d_B, d_C, n);

    // Copy result back to host asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(C, d_C, size, cudaMemcpyDeviceToHost, stream2));

     // Synchronize streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // Verify result
    for (int i = 0; i < n; ++i) {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // Clean up
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));

}
int main() {
    // Your CUDA code here
    int n = 1<<20;
    float *A = (float*)malloc(n*sizeof(float));
    float *B = (float*)malloc(n*sizeof(float));
    float *C = (float*)malloc(n*sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < n; ++i) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }

    vectorAdd(A, B, C, n);
    return 0;
}