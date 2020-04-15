#pragma once
#include <stdexcept>

#define CUDA_CALL(X)          \
  {                           \
    cudaError_t err = X;      \
    if (err != cudaSuccess) { \
      char buf[100] = {0,}; \
      snprintf(buf, 100, "CUDA API error: %d, %s", err, cudaGetErrorString(err)); \
      throw std::runtime_error(buf);              \
    }                         \
  }

#define CUDADRIVER_CALL(func)                          \
  { CUresult err;                                     \
    err = func;                                       \
    if (CUDA_SUCCESS != err) {                        \
      char buf[100] = {0,}; \
      snprintf(buf, 100, "CUDA runtime API error: %d", err); \
      throw std::runtime_error(buf);              \
    }                                                 \
  }

#define MPI_CALL(X)          \
  {                           \
    int err = X;      \
    if (err != MPI_SUCCESS) { \
      char buf[MPI_MAX_ERROR_STRING + 30] = {0,}; \
      char mpierr[MPI_MAX_ERROR_STRING + 1] = {0,}; \
      int len = 0; \
      MPI_Error_string(err, mpierr, &len); \
      snprintf(buf, MPI_MAX_ERROR_STRING + 30, "MPI API error: %d, %s", err, mpierr); \
      throw std::runtime_error(buf);              \
    }                         \
  }


