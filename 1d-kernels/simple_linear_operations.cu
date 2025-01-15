#include <stdio.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>



__global__ void vecAddKernel(float* A, float* B, float* C, int n) // signifies a kernel method
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<n)
    {
        C[i] = A[i] + B[i];
    }
} // core function for data parallelism


__global__ void vecMultKernel(float* A, float* B, float* C, int n) // signifies a kernel method
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<n)
    {
        C[i] = A[i] * B[i];
    }
} // core function for data parallelism


__global__ void vecRandInitialize(float* A, int n, unsigned long seed)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n)
    {
        curandState state;

        // curand_init(seed, sequence_id, offset, &state), Rand num will be same if seed and seq id and offset all are same
        curand_init(seed, i, 0, &state);
        A[i] = curand_uniform(&state);
    }
        // A[i] = (float)rand() / RAND_MAX; rand doesn't work for kernel methods, we need to initialize a seed for each thread.
}


__global__ void ReduceSum(float* A, float* A_cpy, int n, int nearest_power_of_2)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    // int tid = threadIdx.x;

    //check that n is a power of 2 or that A_cpy has memory allocated upto nearest power of 2, waste of mem space so later optimization needed
    if(i<n)
    {
        A_cpy[i] = A[i];
    }
    else if(i < nearest_power_of_2)
    {
        A_cpy[i] = 0;
    }
    __syncthreads();


    for(int stride=nearest_power_of_2/2; stride > 0; stride /= 2 )
    {
        if(i < stride)
        {
            A_cpy[i] += A_cpy[i+stride];
        }
        __syncthreads();
    }
}

void vecInit(float* A, int n)
{
    int size = n*sizeof(float);
    float* d_A;
    
    cudaMalloc((void**)&d_A, size);

    vecRandInitialize<<<ceil(n/256.0), 256>>>(d_A, n, rand() % 69696969);

    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
}


void vecMult(float* A, float* B, float* C, int n)
{
    int size = n*sizeof(float);
    int nearest_power_of_2 = (int)pow(2, ceil(log2(n)));
    float *d_A,*d_B,*d_C,*d_C_cpy;   // d_ implies variable on device (CUDA)

    // Allocate memory in GPU device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_C_cpy, nearest_power_of_2*sizeof(float));

    // Copy data to newly allocated device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    // Perform operations as needed
    vecMultKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
    ReduceSum<<<ceil(n/256.0), 256>>>(d_C, d_C_cpy, n, nearest_power_of_2);

    cudaMemcpy(C, d_C_cpy, size, cudaMemcpyDeviceToHost);
    // Free CUDA memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_cpy);
}


void vecAdd(float* A, float* B, float* C, int n)
{
    int size = n*sizeof(float);
    float *d_A,*d_B,*d_C;   // d_ implies variable on device (CUDA)

    // Allocate memory in GPU device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data to newly allocated device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    // Perform operations as needed
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A,d_B, d_C, n);


    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    // Free CUDA memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


bool checkMult(float* A, float* B, float* C, int n)
{
    const float tolerance = 1e-2;
    float sum = 0.0f;
    for(int i=0; i<n; i++)
    {
        sum += A[i] * B[i];
    }
    printf("Sum: %f\nC[0]: %f\n", sum, C[0]);
    if(fabs(sum - C[0]) > tolerance)
    {
        return false;
    }
    return true;
}
int main()
{
    srand(time(NULL)); // Seed the random number generator
    int n = 4096;
    float* a1 = (float*)malloc(n*sizeof(float));
    float* a2 = (float*)malloc(n*sizeof(float));
    float* a3 = (float*)malloc(n*sizeof(float));
    vecInit(a1,n);
    vecInit(a2,n);
    vecInit(a3,n);

    vecAdd(a1,a2,a3,n);

    for(int i=0;i<10;i++)
    {
        printf("%f + %f = %f\n", a1[i], a2[i], a3[i]);
    }
    vecMult(a1,a2,a3,n);
    printf("Vector Mutiplication Result: %f\n", a3[0]);
    if(checkMult(a1,a2,a3,n))
    {
        printf("Multiplication Success!!\n");
    }
    else
    {
        printf("Multiplication Failure!!\n");
    }
    // printf("Success!!\n");

}