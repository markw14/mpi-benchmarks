#pragma once
#include <stdexcept>

#define CUDA_CALL(X)          \
  {                           \
    cudaError_t err = X;      \
    if (err != cudaSuccess) { \
      char buf[80] = {0,}; \
      snprintf(buf, 80, "CUDA API error: %d, %s", err, cudaGetErrorString(err)); \
      throw std::runtime_error(buf);              \
    }                         \
  }

#define CUDADRIVER_CALL(func)                          \
  { CUresult err;                                     \
    err = func;                                       \
    if (CUDA_SUCCESS != err) {                        \
      char buf[80] = {0,}; \
      snprintf(buf, 80, "CUDA runtime API error: %d", err); \
      throw std::runtime_error(buf);              \
    }                                                 \
  }


