#pragma once

#define CUDA_CALL(X)          \
  {                           \
    cudaError_t err = X;      \
    if (err != cudaSuccess) { \
      throw err;              \
    }                         \
  }

