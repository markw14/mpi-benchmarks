#ifdef WITH_HWLOC

// Inspired by code from: https://www.open-mpi.org/faq/?category=runcuda
#include <string>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include "cuda.h"
#include "mpi.h"
#include "hwloc.h"

#define HWLOC_CALL(call) { int err = call; assert(err == 0); }

struct context {
    hwloc_topology_t topology = nullptr;
    int cnt = 0;
    hwloc_obj_t gpus[16] = { 0, };
    void add(hwloc_obj_t &dev) { gpus[cnt++] = dev; }
};
 
extern void device_set_current(const std::string &pci_id);

static void recursive_find_devices(context &ctx, hwloc_obj_t parent, hwloc_obj_t child) {
    const unsigned nvidia = 0x10de;
    hwloc_obj_t device = hwloc_get_next_child(ctx.topology, parent, child);
    if (device == nullptr) {
        return;
    } else if (device->arity != 0) {
        recursive_find_devices(ctx, device, nullptr);
        recursive_find_devices(ctx, parent, device);
    } else {
        if (device->attr->pcidev.vendor_id == nvidia) {
            //ctx.gpus[ctx.cnt++] = device;
            ctx.add(device);
        }
        recursive_find_devices(ctx, parent, device);
    }
}

#define RETURN_FALSE_IF_NULL(x) { if (x == nullptr) { std::cerr << "FATAL: hwloc_iface: no GPU devices found" << std::endl; return false; } }

bool gpu_conf_init_with_hwloc()
{
    context ctx;
    auto &topology = ctx.topology;
    const unsigned long flags = HWLOC_TOPOLOGY_FLAG_IO_DEVICES | HWLOC_TOPOLOGY_FLAG_IO_BRIDGES;
    //const unsigned long flags = HWLOC_TYPE_FILTER_KEEP_IMPORTANT; -- for hwloc2...
    HWLOC_CALL(hwloc_topology_init(&topology));
    HWLOC_CALL(hwloc_topology_set_flags(topology, flags));
    HWLOC_CALL(hwloc_topology_load(topology));
    hwloc_cpuset_t cset = hwloc_bitmap_alloc();
    HWLOC_CALL(hwloc_get_last_cpu_location(topology, cset, 0));
    hwloc_obj_t n = hwloc_get_first_largest_obj_inside_cpuset(topology, cset);
    while (n != nullptr && n->type != HWLOC_OBJ_NODE) { // on hwloc2 API it fails just here
        n = n->parent;
    }
    RETURN_FALSE_IF_NULL(n);
    hwloc_obj_t br = hwloc_get_next_child(topology, n, nullptr);
    while (br != nullptr && br->type != HWLOC_OBJ_BRIDGE) {
        br = hwloc_get_next_child(topology, n, br);
    }
    RETURN_FALSE_IF_NULL(br);
    recursive_find_devices(ctx, br, nullptr);
    if (ctx.gpus[0] == 0) {
         RETURN_FALSE_IF_NULL(nullptr);
    } else {
        // NOTE: we always go with device 0! This is not OK if we have more than 1 device per
        // NUMA node. The problem is that we have no information to understand which device to pick.
        const size_t device_num = 0;
        char pci_bus_id[16];
        sprintf(pci_bus_id, "%.2x:%.2x:%.2x.%x", ctx.gpus[device_num]->attr->pcidev.domain, ctx.gpus[device_num]->attr->pcidev.bus,
                                                 ctx.gpus[device_num]->attr->pcidev.dev, ctx.gpus[device_num]->attr->pcidev.func);
        device_set_current(pci_bus_id);
    }
    return true;
}
#endif
