#include <iostream>
#include "cuda_helpers.h"
#include "device_routines.h"
#include <cuda.h>
#include <assert.h>

static cudaStream_t stream_main = 0, stream_workload = 0;
static cudaEvent_t event = 0;
static bool initialized = false;

void device_init_contexts()
{
    CUDA_CALL(cudaStreamCreateWithFlags(&stream_main, cudaStreamNonBlocking))
    CUDA_CALL(cudaStreamCreateWithFlags(&stream_workload, cudaStreamNonBlocking))
    CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
}

void device_sync_context()
{
    CUDA_CALL(cudaStreamSynchronize(stream_main));
    CUDA_CALL(cudaStreamSynchronize(stream_workload));
}

size_t device_get_num_of_dev()
{
    int n;
    CUDA_CALL(cudaGetDeviceCount(&n));
    return n;
}

void device_set_current(size_t n)
{
    assert(!initialized);
    std::cout << "GPU device set: cuda_id=" << n << std::endl;
    CUDA_CALL(cudaSetDevice(n));
    device_init_contexts();
    initialized = true;
}

void device_set_current(const std::string &pci_id)
{
    assert(!initialized);
    CUdevice dev;
    char devname[256];
    CUDADRIVER_CALL(cuInit(0));
    CUDADRIVER_CALL(cuDeviceGetByPCIBusId(&dev, pci_id.c_str()));
    CUDADRIVER_CALL(cuDeviceGetName(devname, 256, dev));
    std::cout << "GPU device set: pci_id=" << pci_id << ", name=" << devname << " (with hwloc)" << std::endl;
    initialized = true;
}

bool device_is_idle()
{
    if (event) {
        CUDA_CALL(cudaEventRecord(event, stream_workload));
        cudaError_t ret = cudaEventQuery(event);
        if (ret != cudaErrorNotReady && ret != cudaSuccess) {
            // error case: throw exception
            CUDA_CALL(ret);
        }
        if (ret == cudaErrorNotReady) {
            // stream has some load currently, not idle
            return false;
        }
    }
    return true;
}

template <int SIZE>
__global__ void workload(int ncycles, int CALIBRATION_CONST) {
    __shared__ double a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE];
    while (ncycles--) {
        for (int N = 0; N < CALIBRATION_CONST; N++) {
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    for (int k = 0; k < SIZE; k++) {
                        c[i][j] += a[i][k] * b[k][j] + N * N;
                    }
                }
            }
        }
    }
}

void device_submit_workload(int ncycles, int calibration_const)
{
    constexpr int array_dim = 10;
    workload<array_dim><<<1, 1, 0, stream_workload>>>(ncycles, calibration_const);
}

void d2h_transfer(char *to, char *from, size_t size, transfer_t type)
{
    CUDA_CALL(cudaMemcpyAsync(to, from, size, cudaMemcpyDeviceToHost, 
                              type == transfer_t::MAIN ? stream_main : stream_workload));
    if (type == transfer_t::MAIN) {
        CUDA_CALL(cudaStreamSynchronize(stream_main))
    }
}

void h2d_transfer(char *to, char *from, size_t size, transfer_t type)
{
    CUDA_CALL(cudaMemcpyAsync(to, from, size, cudaMemcpyHostToDevice, 
                              type == transfer_t::MAIN ? stream_main : stream_workload));
    if (type == transfer_t::MAIN) {
        CUDA_CALL(cudaStreamSynchronize(stream_main))
    }
}
