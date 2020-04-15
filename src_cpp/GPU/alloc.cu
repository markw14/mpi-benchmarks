#include "alloc.h"
#include "cuda_helpers.h"

extern void mpi_host_mem_alloc(char *&host_buf, size_t size_to_alloc);
extern void mpi_host_mem_free(char *&host_buf);
void host_mem_alloc(char *&host_buf, size_t size_to_alloc, host_alloc_t atype)
{
    switch (atype) {
        case ALLOC_CUDA: CUDA_CALL(cudaHostAlloc(&host_buf, size_to_alloc, 
                                                 cudaHostAllocPortable));
                         break;
        case ALLOC_MPI: mpi_host_mem_alloc(host_buf, size_to_alloc);
                        CUDA_CALL(cudaHostRegister(host_buf, size_to_alloc, 
                                  cudaHostRegisterPortable));
                        break;
        default: throw std::runtime_error("Unknown memory allocation mode");
    }
    return;
}

void host_mem_free(char *&host_buf, host_alloc_t atype)
{
    switch (atype) {
        case ALLOC_CUDA: CUDA_CALL(cudaFreeHost(host_buf));
                         break;
        case ALLOC_MPI: CUDA_CALL(cudaHostUnregister(host_buf));
                        mpi_host_mem_free(host_buf);
                        break;
        default: throw std::runtime_error("Unknown memory allocation mode");
    }
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

