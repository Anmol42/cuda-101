#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define D2D cudaMemcpyDeviceToDevice
#define H2H cudaMemcpyHostToHost

__global__ void rowSwap(float *A, int row1, int row2, int m, int n)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n)
    {
        int temp = A[row1*n + i];
        A[row1*n + i] = A[row2*n + i];
        A[row2*n + i] = temp;
    }
}


void init_matrix(float* mat, int rows, int cols)
{
    for(int i=0; i<rows*cols; i++)
    {
        mat[i] = rand() % 10;
    }
}


int main()
{
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    int cores_per_sm = prop.multiProcessorCount * 128; //_ConvertSMVer2Cores(prop.major, prop.minor); we don't have the necessary helper header so hardcoding
    float clock_rate_ghz = prop.clockRate / 1e6f;  // Convert from kHz to GHz
    float theoretical_gflops = cores_per_sm * clock_rate_ghz * 2.0f;

    float mem_clock_ghz = prop.memoryClockRate / 1e6f;  // Convert kHz to GHz
    float mem_bus_width_bytes = prop.memoryBusWidth / 8.0f;  // Convert bits to bytes
    float theoretical_bw = mem_clock_ghz * mem_bus_width_bytes * 2;  // Factor 2 for DDR

    printf("Theoretical Peak GFLOPS: %.2f GFLOPS\n", theoretical_gflops);
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", theoretical_bw);

    int n = 2048;
    size_t size = n*(n+1)*sizeof(float);
    float *A = (float*) malloc(size);
    init_matrix(A,n, n+1);
    return EXIT_SUCCESS;
}