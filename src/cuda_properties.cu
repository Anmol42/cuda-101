#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


bool checkSharedMemoryLimits(int tileWidth) {
    cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, 0);
    
    if (error != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(error));
        return false;
    }

    // Calculate required shared memory for one block
    size_t requiredSharedMemPerBlock = 5 * tileWidth * tileWidth * sizeof(float);
    
    printf("Device: %s\n", deviceProp.name);
    printf("Shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Required shared memory: %zu bytes\n", requiredSharedMemPerBlock);
    
    if (requiredSharedMemPerBlock > deviceProp.sharedMemPerBlock) {
        printf("Error: Tile size too large! Would require %zu bytes of shared memory but only %zu available\n",
               requiredSharedMemPerBlock, deviceProp.sharedMemPerBlock);
        return false;
    }
    
    // Check maximum threads per block
    int threadsPerBlock = tileWidth * tileWidth;
    if (threadsPerBlock > deviceProp.maxThreadsPerBlock) {
        printf("Error: Tile size requires %d threads but device maximum is %d\n",
               threadsPerBlock, deviceProp.maxThreadsPerBlock);
        return false;
    }
    
    return true;
}


int main()
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("%d\n", dev_count);
    cudaDeviceProp prop;
    for (int i = 0; i < dev_count; i++) 
    {
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  CUDA Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %lu bytes\n", prop.totalGlobalMem);
        printf("  Shared Memory Per Block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers Per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads Dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Clock Rate: %d kHz\n", prop.clockRate);
        printf("  Total Constant Memory: %lu bytes\n", prop.totalConstMem);
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("  Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
        printf("  Max Threads Per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("  Managed Memory: %s\n", prop.managedMemory ? "Yes" : "No");
        printf("\n");
        checkSharedMemoryLimits(32);
    }
    return EXIT_SUCCESS;
}
