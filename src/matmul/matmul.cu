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

    double M = m, N = n, K = k;
    float total_operations = 2*M*N*K;
    float flops = total_operations / (milliseconds / 1000.0f);  // Total FLOPS
    float gflops = flops / 1e9;  // Convert to GFLOPS

    printf("Achieved GFLOPS: %f GFLOPS\n", gflops);

    double bytes_transferred_real = (2*M*N*K + M*N)*sizeof(float);
    double achieved_bw = (bytes_transferred_real / (milliseconds / 1000.0f)) / 1e9;

    printf("Achieved Memory Bandwidth: %.2f GB/s, %f\n", achieved_bw, bytes_transferred_real);

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


void write_matrix_to_csv(float* matrix, int rows, int cols, const char* filename)
{
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file for writing");
        return;
    }


    // Write the matrix values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%lf", matrix[i*cols+j]);
            if (j < cols - 1) {
                fprintf(file, ","); // Add comma between values in a row
            }
        }
        fprintf(file, "\n"); // Newline after each row
    }

    fclose(file);
    printf("Matrix written to %s successfully.\n", filename);
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


int main()
{
    srand(time(NULL));
    cudaFuncSetCacheConfig(matMulKernel, cudaFuncCachePreferNone);
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
    A = read_matrix_from_csv("./src/matmul/input_A_int.csv", &m, &k, ',');
    B = read_matrix_from_csv("./src/matmul/input_B_int.csv", &k, &n, ',');
    actual_result = read_matrix_from_csv("./src/matmul/output_int.csv", &m, &n);
    C = (float*)malloc(m * n * sizeof(float));

    matMul(A, B, C, m, k, n);
    matMulAccuracy(C, actual_result, m, n);


    return EXIT_SUCCESS;
}