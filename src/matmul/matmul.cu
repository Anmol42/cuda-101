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



/**
 * @brief Matrix multiplication kernel.
 *
 * This kernel performs matrix multiplication of two matrices A and B, storing the result in matrix C.
 * The matrices are in row-major order.
 *
 * @param A Pointer to the first input matrix (m x k).
 * @param B Pointer to the second input matrix (k x n).
 * @param C Pointer to the output matrix (m x n).
 * @param m Number of rows in matrix A and matrix C.
 * @param k Number of columns in matrix A and number of rows in matrix B.
 * @param n Number of columns in matrix B and matrix C.
 *
 * The kernel uses a simple row-wise and column-wise approach to compute the matrix product.
 * Each thread computes one element of the output matrix C.
 * The ratio of floating-point operations to global memory accesses is 1.0.
 * Optimizations such as tiling and shared memory usage can improve this ratio and overall performance.
 */
__global__ void matMulKernel(float* A, float* B, float* C, int m, int k, int n)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    if(row < m && col < n)
    {
        int offset = col + row*n;
        float sum = 0;
        for(int i=0;i<k;i++)
        {
            sum += A[i + k*row]*B[col + i*n];
        }
        C[offset] = sum;
    }
}

void matMul(float* h_A, float* h_B, float* h_C, int m, int k, int n)
{
    cudaEvent_t start, stop;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // A is m x k, B is k x n, C is m x n
    int size_A = m*k*sizeof(float);
    int size_B = k*n*sizeof(float);
    int size_C = m*n*sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, HostToDevice);
    cudaMemcpy(d_B, h_B, size_B, HostToDevice);
    // cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // call kernel, if direct integer division is used, it will be 0 and hence ceil will return 0 as truncation occured first.
    dim3 dimGrid(ceil(n/16.0), ceil(m/16.0), 1);
    dim3 dimBlock(16,16,1);
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("MatMul kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(h_C, d_C, size_C, DeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
    int m=10240,k=10240,n=10240;
    float *A,*B,*C;
    A = (float*)malloc(m * k * sizeof(float));
    B = (float*)malloc(k * n * sizeof(float));
    C = (float*)malloc(m * n * sizeof(float));

    init_matrix(A, m, k);
    init_matrix(B, k, n);

    clock_t start_time, end_time;
    start_time = clock();
    matMul(A, B, C, m, k, n);
    end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Time taken: %f seconds\n", time_taken);


    return EXIT_SUCCESS;
}