#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define DeviceToHost cudaMemcpyDeviceToHost
#define HostToDevice cudaMemcpyHostToDevice
#define HostToHost cudaMemcpyHostToHost
#define DeviceToDevice cudaMemcpyDeviceToDevice 



//Tiled matrix multiplication
__global__ void tiledMatMulKernel(float* A, float* B, float* result, int m, int k, int n)
{
    const int TILE_WIDTH = 64;
    // TILE_WIDTH must be known at compile time for this to work
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    // int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    //global row and column is given here
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float value = 0;

    //load tile into shared memory
    for(int i=0;i<(k+TILE_WIDTH-1)/TILE_WIDTH;i++)
    {
        if(row < m && (TILE_WIDTH*i+tx) < k)
            Ads[ty][tx] = A[row*k+TILE_WIDTH*i+tx];
        else
            Ads[ty][tx] = 0.0f;
        if(col < n && (TILE_WIDTH*i+ty) < k)
            Bds[ty][tx] = B[col+n*(TILE_WIDTH*i+ty)];
        else
            Bds[ty][tx] = 0.0f;
        __syncthreads();

        for(int j=0;j<TILE_WIDTH;j++)
        {
            value += __fmaf_rn(Ads[ty][j],Bds[j][tx], 0.0f);
            __threadfence_block();
        }
        __syncthreads();
    }
    if(row<m && col<n)
        result[row*n+col] = value;

}


void matMul(float* h_A, float* h_B, float* h_C, int m, int k, int n)
{
    cudaEvent_t start, stop;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int TILE_WIDTH = 64;
    // A is m x k, B is k x n, C is m x n
    int size_A = m*k*sizeof(float);
    int size_B = k*n*sizeof(float);
    int size_C = m*n*sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaEventRecord(start);
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for memory allocation on GPU: %f ms\n", milliseconds);

    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size_A, HostToDevice);
    cudaMemcpy(d_B, h_B, size_B, HostToDevice);
    cudaEventRecord(stop);

    // Memcpy is a sync function so calling a synchronise is not needed
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for copying host data on GPU: %f ms\n", milliseconds);

    // call kernel, if direct integer division is used, it will be 0 and hence ceil will return 0 as truncation occured first.
    dim3 dimGrid(ceil(n/(float)(TILE_WIDTH)), ceil(m/(float)(TILE_WIDTH)), 1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
   
    tiledMatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiled MatMul kernel execution time: %f ms\n", milliseconds);
    double M = m, N = n, K = k;
    float total_operations = 2*M*N*K;
    float flops = total_operations / (milliseconds / 1000.0f);  // Total FLOPS
    float gflops = flops / 1e9;  // Convert to GFLOPS

    printf("Achieved GFLOPS: %f GFLOPS\n", gflops);

    double bytes_transferred = (2*M*N*K/(float)(TILE_WIDTH) + M*N)*sizeof(float);
    double achieved_bw = (bytes_transferred / (milliseconds / 1000.0f)) / 1e9;

    printf("Achieved Memory Bandwidth: %.2f GB/s, %f\n", achieved_bw, bytes_transferred);

    cudaMemcpy(h_C, d_C, size_C, DeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


void init_matrix(float* mat, int rows, int cols)
{
    for(int i=0; i<rows*cols; i++)
    {
        mat[i] = -1000.0f + (static_cast<float>(rand()) / RAND_MAX) * 2000.0f;
    }
}


int main()
{
    srand(time(NULL));
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

    int m=1024,k=1024,n=1024;
    float *A,*B,*C;
    A = (float*)malloc(m * k * sizeof(float));
    B = (float*)malloc(k * n * sizeof(float));
    C = (float*)malloc(m * n * sizeof(float));

    init_matrix(A, m, k);
    init_matrix(B, k, n);

    matMul(A, B, C, m, k, n);


    return  EXIT_SUCCESS;
}
