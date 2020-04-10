#if 0
#include "iostream"
#include "cuda_helpers.h"

void d2h_transfer(char *to, char *from, size_t size)
{
    CUDA_CALL(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
}

void h2d_transfer(char *to, char *from, size_t size)
{
    CUDA_CALL(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
}
#endif
