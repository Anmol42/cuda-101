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


__global__ void sgemm_naive(float* A, float* B, float* C, int m, int k, int n)
{
    const uint row = threadIdx.y + blockIdx.y*blockDim.y;
    const uint col = threadIdx.x + blockDim.x*blockIdx.x;

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


__global__ void sgemm_coalesced(float* A, float* B, float* C, int m, int k, int n)
{

}


void matmul(float* A, float* B, float* C, int m, int k, int n)
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

    dim3 dimGrid((n+TILE_WIDTH-1)/(TILE_WIDTH), (m+TILE_WIDTH-1)/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
    printf("Grid size: %dx%d\n", (n+TILE_WIDTH-1)/(TILE_WIDTH), (m+TILE_WIDTH-1)/TILE_WIDTH);


    cudaDeviceSynchronize();
    cudaEventRecord(start);
   
    sgemm_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);       // replace_kernels here
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    double M = m, N = n, K = k;
    float total_operations = 2*M*N*K;
    float flops = total_operations / (milliseconds / 1000.0f);  // Total FLOPS
    float gflops = flops / 1e9;  // Convert to GFLOPS

    printf("Achieved GFLOPS: %f GFLOPS\n", gflops);

    double bytes_transferred_real = (2*M*N*K + M*N)*sizeof(float);
    double achieved_bw = (bytes_transferred_real / (milliseconds / 1000.0f)) / 1e9;

    printf("Achieved Memory Bandwidth: %.2f GB/s\n", achieved_bw);

    cudaMemcpy(C, d_C, size_C, D2H);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

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

    // init_matrix(A, m, k);
    // init_matrix(B, k, n);

    matmul(A, B, C, m, k, n);
    matMulAccuracy(C, actual_result, m, n); // transfer this output to a temporary file for better analysis

    return  EXIT_SUCCESS;
}
