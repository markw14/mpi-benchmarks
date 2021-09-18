#pragma once

#include <unistd.h>
#include <climits>
#include <math.h>

#if 1
namespace sys {
// NOTE: seems to be Linux-specific
static inline size_t getnumcores() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

static inline bool threadaffinityisset(int &nthreads) {
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        assert(false && "sched_getaffinity failure");
    }
    int NC = sys::getnumcores();
    int nset = 0;
    for (int i = 0; i < NC; i++) {
        nset += (CPU_ISSET(i, &mask) ? 1 : 0);
    }
    nthreads = nset;
    // We assume OK: exact one-to-one affinity or hyperthreading/SMT affinity
    // for 2, 3 or 4 threads
    return nthreads > 0 && nthreads < 5 && nthreads != NC;
}

static inline int getthreadaffinity() {
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        assert(false && "sched_getaffinity failure");
    }
    int core = -1;
    for (size_t i = 0; i < sizeof(mask) * 8; i++) {
        if (CPU_ISSET(i, &mask)) {
            core = (int)i;
            break;
        }
    }
    assert(core != -1);
    assert(core < (int)sys::getnumcores());
    return core;
}
}
#endif


