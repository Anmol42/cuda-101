#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    }
    return EXIT_SUCCESS;
}