#include <iostream>
#include "cuda_helpers.h"
#include "device_routines.h"

//static cudaStream_t stream_main, stream_workload;

void device_init_contexts()
{
    // TODO create streams here
}

size_t device_get_num_of_dev()
{
    int n;
    CUDA_CALL(cudaGetDeviceCount(&n));
    return n;
}

void device_set_current(size_t n)
{
    std::cout << "GPU device set: " << n << std::endl;
    CUDA_CALL(cudaSetDevice(n));    
}

void device_set_current(std::string pci_id)
{
//    std::cout << "GPU device set: " << n << std::endl;
//    CUDA_CALL(cudaSetDevice(n));    
}

char *device_alloc_mem(size_t size)
{
    //cudaMalloc
    //cudaMemset
    return nullptr;
}

void device_free_mem(char *ptr)
{
    //cudaFree
}

bool device_is_idle()
{
    //
    return true;
}

void device_submit_workload(int ncycles)
{
    // TODO appropriate stream
}

void d2h_transfer(char *to, char *from, size_t size, transfer_t type)
{
    // TODO do async transfer, select stream with type
    CUDA_CALL(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
    // TODO for type == MAIN, do cudaStreamSync
}

void h2d_transfer(char *to, char *from, size_t size, transfer_t type)
{
    // TODO do async transfer, select stream with type
    CUDA_CALL(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
    // TODO for type == MAIN, do cudaStreamSync
}
