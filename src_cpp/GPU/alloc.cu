#include "cuda_helpers.h"

void host_mem_alloc(char *&host_buf, size_t size_to_alloc)
{
    CUDA_CALL(cudaHostAlloc(&host_buf, size_to_alloc, cudaHostAllocPortable))
    return;
}

void host_mem_free(char *&host_buf)
{
    CUDA_CALL(cudaFreeHost(host_buf));
    return;
}

void device_mem_alloc(char *&device_buf, size_t size_to_alloc)
{
    CUDA_CALL(cudaMalloc(&device_buf, size_to_alloc));
    CUDA_CALL(cudaMemset(device_buf, 0, size_to_alloc));
    return;
}

void device_mem_free(char *&device_buf)
{
    if (device_buf) {
        CUDA_CALL(cudaFree(device_buf));
    }
    return;
}

