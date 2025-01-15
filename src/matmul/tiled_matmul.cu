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


    // These indices need TILE_WIDTH to be equal to blockDim.x/y for it to work (which is a mandatory condition almost always)
    // int Row = by * TILE_WIDTH + ty;
    // int Col = bx * TILE_WIDTH + tx;

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
            value += Ads[ty][j]*Bds[j][tx];
        }
        __syncthreads();
    }
    if(row<m && col<n)
        result[row*n+col] = value;

}


void matMul(float* h_A, float* h_B, float* h_C, int m, int k, int n)
{
    const int TILE_WIDTH = 64;
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
    dim3 dimGrid(ceil(n/float(TILE_WIDTH)), ceil(m/float(TILE_WIDTH)), 1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
    tiledMatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);

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
    int m=10240,k=102400,n=10240;
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

    // Print the result
    printf("Time taken: %f seconds\n", time_taken);
    return  EXIT_SUCCESS;
}
