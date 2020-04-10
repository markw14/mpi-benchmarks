#include "cuda_helpers.h"

void device_mem_alloc(char *&device_buf, size_t size_to_alloc)
{
#if 1
    CUDA_CALL(cudaMalloc(&device_buf, size_to_alloc));
#else
    device_buf = ;
#endif
    return;
}

void host_mem_alloc(char *&host_buf, size_t size_to_alloc)
{
#if 1
    host_buf = (char *)calloc(size_to_alloc, 1);
#else
    host_buf = ;
#endif
    return;
}

void device_mem_free(char *&device_buf)
{
#if 1
    if (device_buf) {
        CUDA_CALL(cudaFree(device_buf));
    }
#else
    device_buf = ;
#endif
    return;
}

void host_mem_free(char *&host_buf)
{
#if 1
    free(host_buf);
#else
    host_buf = ;
#endif
    return;
}

