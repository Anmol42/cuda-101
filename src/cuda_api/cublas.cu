#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define D2D cudaMemcpyDeviceToDevice
#define H2H cudaMemcpyHostToHost

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


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


int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    return EXIT_SUCCESS;
}