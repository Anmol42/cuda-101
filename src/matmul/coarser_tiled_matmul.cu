#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define H2H cudaMemcpyHostToHost
#define D2D cudaMemcpyDeviceToDevice 

#define TILE_WIDTH 32


/*
 *
 * This code implements a matrix multiplication using a coarser threading approach in CUDA.
 * By giving up a bit of parallelism from the tiled matrix multiplication, it aims to reduce 
 * redundant read accesses. This approach can help in optimizing memory access patterns and 
 * potentially improve performance by minimizing the number of redundant reads from global memory.
 *
 */


// Coarser matrix multiplication
__global__ void coarse_tiled_matmul_kernel(float* A, float* B, float* C, int m, int k, int n)
{
    int col = blockDim.x*TILE_WIDTH + threadIdx.x;
    int row = blockDim.y*TILE_WIDTH + threadIdx.y;
}



float* read_matrix_from_csv(const char* filename, int *rows, int *cols, char delimiter=',')
{
    FILE* file = fopen(filename, "r");
    if(file == NULL)
    {
        printf("Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    size_t capacity = 1024;  // Initial capacity for the buffer
    float* buffer = (float*)malloc(capacity * sizeof(float));
    if (!buffer) 
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t line_capacity = 1024; // Initial line buffer capacity
    char* line = (char*)malloc(line_capacity * sizeof(char));
    if (!line) 
    {
        fprintf(stderr, "Error: Memory allocation for line buffer failed\n");
        free(buffer);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    *rows = 0;
    *cols = 0;
    int current_cols = 0;

    while (getline(&line, &line_capacity, file) != -1)  // Dynamically read each line
    {
        current_cols = 0;

        // Parse the line based on the delimiter
        char* token = strtok(line, &delimiter);
        while (token) 
        {
            if (*rows == 0) (*cols)++; // Count columns based on the first row

            if ((*rows) * (*cols) + current_cols >= capacity) 
            {
                capacity *= 2;
                buffer = (float*)realloc(buffer, capacity * sizeof(float)); // realloc copies the data for you
                if (!buffer) 
                {
                    fprintf(stderr, "Error: Memory reallocation failed\n");
                    free(line);
                    fclose(file);
                    exit(EXIT_FAILURE);
                }
            }

            buffer[(*rows) * (*cols) + current_cols] = strtof(token, NULL);
            current_cols++;
            token = strtok(NULL, &delimiter);
        }

        if (*rows > 0 && current_cols != *cols) 
        {
            fprintf(stderr, "Error: Inconsistent number of columns in row %d\n", *rows + 1);
            free(buffer);
            free(line);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        (*rows)++;
    }

    fclose(file);
    free(line);

    // Resize buffer to the exact size
    buffer = (float*)realloc(buffer, (*rows) * (*cols) * sizeof(float));
    if (!buffer) 
    {
        fprintf(stderr, "Error: Memory reallocation failed\n");
        exit(EXIT_FAILURE);
    }

    return buffer;
}


void matMulAccuracy(float *result, float* expected_result, int rows, int cols)
{
    int cnt = 0;
    float err = 1e-2;
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            if(abs((result[i*cols+j] - expected_result[i*cols+j])/expected_result[i*cols+j]) > err)
            {
                cnt++;
                // printf("Error at index (%d, %d): %f != %f\n", i, j, result[i*cols+j], expected_result[i*cols+j]);
            }
        }
    }
    printf("Number of elements that differ by relative error more than %f: %d out of %ld\n", err, cnt, (long int)rows*cols);
}


void coarse_tiled_matmul(float* A, float* B, float* C, int m, int k, int n)
{
    cudaEvent_t start, stop;
    float milliseconds;
    float *d_A, *d_B, *d_C;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t size_A = (long int)m*k*sizeof(float);
    size_t size_B = (long int)k*n*sizeof(float);
    size_t size_C = (long int)m*n*sizeof(float);

    cudaEventRecord(start);
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for memory allocation on GPU: %f ms\n", milliseconds);

    cudaEventRecord(start);
    cudaMemcpy(d_A, A, size_A, H2D);
    cudaMemcpy(d_B, B, size_B, H2D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for copying host data on GPU: %f ms\n", milliseconds);


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

    int m,k,n;
    float *A,*B,*C,*actual_result;
    A = read_matrix_from_csv("./src/matmul/input_A.csv", &m, &k, ',');
    B = read_matrix_from_csv("./src/matmul/input_B.csv", &k, &n, ',');
    actual_result = read_matrix_from_csv("./src/matmul/output.csv", &m, &n);
    C = (float*)malloc(m * n * sizeof(float));

    return EXIT_SUCCESS;
}