#ifdef WITH_HWLOC
/**
 * Test program to show the use of hwloc to select the GPU closest to the CPU
 * that the MPI program is running on.  Note that this works even without
 * any libpciacces or libpci support as it keys off the NVIDIA vendor ID.
 * There may be other ways to implement this but this is one way.
 * January 10, 2014
 */
#include <assert.h>
#include <stdio.h>
#include "cuda.h"
#include "mpi.h"
#include "hwloc.h"
 
#define ABORT_ON_ERROR(func)                          \
  { CUresult res;                                     \
    res = func;                                       \
    if (CUDA_SUCCESS != res) {                        \
        printf("%s returned error=%d\n", #func, res); \
        abort();                                      \
    }                                                 \
  }

static hwloc_topology_t topology = NULL;
static int gpuIndex = 0;
static hwloc_obj_t gpus[16] = {0};
 
/**
 * This function searches for all the GPUs that are hanging off a NUMA
 * node.  It walks through each of the PCI devices and looks for ones
 * with the NVIDIA vendor ID.  It then stores them into an array.
 * Note that there can be more than one GPU on the NUMA node.
 */
 
static void find_gpus(hwloc_topology_t topology, hwloc_obj_t parent, hwloc_obj_t child) {
    hwloc_obj_t pcidev;
    pcidev = hwloc_get_next_child(topology, parent, child);
    if (NULL == pcidev) {
        return;
    } else if (0 != pcidev->arity) {
        /* This device has children so need to look recursively at them */
        find_gpus(topology, pcidev, NULL);
        find_gpus(topology, parent, pcidev);
    } else {
        if (pcidev->attr->pcidev.vendor_id == 0x10de) {
            gpus[gpuIndex++] = pcidev;
        }
        find_gpus(topology, parent, pcidev);
    }
}

bool gpu_conf_init_with_hwloc()
{
    const unsigned long flags = HWLOC_TOPOLOGY_FLAG_IO_DEVICES | HWLOC_TOPOLOGY_FLAG_IO_BRIDGES;
    //const unsigned long flags = HWLOC_TYPE_FILTER_KEEP_IMPORTANT; -- for hwloc2...
    hwloc_cpuset_t newset;
    hwloc_obj_t node, bridge;
    char pciBusId[16];
    CUdevice dev;
    char devName[256];

    /* Now decide which GPU to pick.  This requires hwloc to work properly.
     * We first see which CPU we are bound to, then try and find a GPU nearby.
     */
    int retval = hwloc_topology_init(&topology);
    assert(retval == 0);
    retval = hwloc_topology_set_flags(topology, flags);
    assert(retval == 0);
    retval = hwloc_topology_load(topology);
    assert(retval == 0);
    newset = hwloc_bitmap_alloc();
    retval = hwloc_get_last_cpu_location(topology, newset, 0);
    assert(retval == 0);
 
    /* Get the object that contains the cpuset */
    node = hwloc_get_first_largest_obj_inside_cpuset(topology, newset);
 
    /* Climb up from that object until we find the HWLOC_OBJ_NODE */
    while (node->type != HWLOC_OBJ_NODE) { // on hwloc2 API it fails just here
        node = node->parent;
    }
 
    /* Now look for the HWLOC_OBJ_BRIDGE.  All PCI busses hanging off the
     * node will have one of these */
    bridge = hwloc_get_next_child(topology, node, NULL);
    while (bridge->type != HWLOC_OBJ_BRIDGE) {
        bridge = hwloc_get_next_child(topology, node, bridge);
    }
 
    /* Now find all the GPUs on this NUMA node and put them into an array */
    find_gpus(topology, bridge, NULL);
 
    ABORT_ON_ERROR(cuInit(0));
    /* Now select the first GPU that we find */
    if (gpus[0] == 0) {
        printf("FATAL: no GPU devices found\n");
        return false;
    } else {
        sprintf(pciBusId, "%.2x:%.2x:%.2x.%x", gpus[0]->attr->pcidev.domain, gpus[0]->attr->pcidev.bus,
        gpus[0]->attr->pcidev.dev, gpus[0]->attr->pcidev.func);
        ABORT_ON_ERROR(cuDeviceGetByPCIBusId(&dev, pciBusId));
        ABORT_ON_ERROR(cuDeviceGetName(devName, 256, dev));
        printf("GPU device set: GPU=%s, name=%s (with hwloc)\n", pciBusId, devName);
    }
    return true;
}
#endif
