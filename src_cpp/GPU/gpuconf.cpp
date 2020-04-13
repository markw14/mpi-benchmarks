#include <unistd.h>
#include <strings.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <stdint.h>


struct gpu_conf {
    int ncores = 0, ngpus = 0;
    std::map<int, int> core_to_gpu;
    std::vector<int> cores;
    mutable int core = -1;
    int nthreads = 0;
    void init_generic(); // gets ncores and ngpus from system 
    void init_from_str(const std::string &str);
    int gpu_by_core(int core) const {
        auto gpuit = core_to_gpu.find(core);
        assert(gpuit != core_to_gpu.end());
        return gpuit->second;
    }
};

size_t device_get_num_of_dev();
void device_set_current(size_t n);

// THIS block may have portability issues (POSIX or even Linux specifics)
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
    assert (core != -1);
    assert(core < (int)sys::getnumcores());
    return core;
}
}
#endif

namespace helpers {
static inline void str_split(std::string s, char delimiter, 
                             std::vector<std::string>& result) {
    result.clear();
    std::string token;
    std::istringstream token_stream(s);
    while (std::getline(token_stream, token, delimiter)) {
        result.push_back(token);
    }
}

static inline void vstr_to_vint(std::vector<std::string>& from, 
                                std::vector<int>& to) {
    to.clear();
    for (auto& s : from) {
        int x = std::stoi(s);
        to.push_back(x);
    }
}
}
void gpu_conf::init_generic() {
    core_to_gpu.clear();
    size_t NC = sys::getnumcores();
    size_t NG = device_get_num_of_dev();
    for (size_t i = 0; i < NC; i++) {
        int G = -1;
        if (NG)
            G = i * NG / NC;
        core_to_gpu[i] = G;
    }
    ncores = NC;
    ngpus = NG;
    return;
}

void gpu_conf::init_from_str(const std::string &str) {
    if (str.empty()) {
        init_generic();
        return;
    }
    try {
        std::vector<std::string> s_numas;
        helpers::str_split(str, ';', s_numas);
        size_t ngpus = 0;
        int numa = 0;
        for (auto& s_numa : s_numas) {
            std::vector<std::string> s_core_gpu;
            std::vector<std::string> s_gpus;
            std::vector<int> gpus;
            helpers::str_split(s_numa, '@', s_core_gpu);
            if (s_core_gpu.size() == 2) {
                helpers::str_split(s_core_gpu[1], ',', s_gpus);
                helpers::vstr_to_vint(s_gpus, gpus);
                assert(s_gpus.size() == gpus.size());
            }
            std::vector<std::string> s_cores;
            std::vector<int> local_cores;
            helpers::str_split(s_core_gpu[0], ',', s_cores);
            if (s_cores.size() == 1 && (!strncasecmp(s_cores[0].c_str(), "0x", 2))) {
                uint64_t mask = 0;
                long long n = std::stoll(s_cores[0], nullptr, 16);
                assert(n > 0 && n < (long long)UINT64_MAX + 1);
                mask = (uint64_t)n;
                for (int j = 0; j < 64; j++) {
                    if ((uint64_t)mask & ((uint64_t)1 << j)) {
                        local_cores.push_back(j);
                    }
                }
            } else {
                helpers::vstr_to_vint(s_cores, local_cores);
                assert(s_cores.size() == local_cores.size());
            }
            size_t NC = local_cores.size();
            size_t NG = gpus.size();
            for (size_t i = 0; i < NC; i++) {
                int G = -1;
                if (NG)
                    G = gpus[i * NG / NC];
                core_to_gpu[local_cores[i]] = G;
            }
            numa++;
            ngpus += NG;
            ncores += NC;
            cores.insert(cores.end(), local_cores.begin(), local_cores.end());
        }
    } catch (std::runtime_error& ex) {
        std::cout << std::string("gpuconf: handling/parsing conf string failed,"
                                 " falling back to generic: ") + ex.what() << std::endl;
        init_generic();
        return;
    } catch (...) {
        std::cout << "gpuconf: handling/parsing conf string,"
                     " falling back to generic." << std::endl;
        init_generic();
        return;
    }
}

bool gpu_conf_init(const std::string &str)
{
    gpu_conf conf;
    conf.init_from_str(str);
    if (device_get_num_of_dev() == 0) {
        std::cout << "FATAL: no GPU devices found" << std::endl;
        return false;
    }
    int nthreads = 0;
    if (!sys::threadaffinityisset(nthreads)) {
        std::cout << "WARNING: thread affinity seems to be not set,"
                     " can't choose relevant GPU device" << std::endl;
        return true;
    }
    int gpu = conf.gpu_by_core(sys::getthreadaffinity());
    assert(gpu != -1);
    device_set_current(gpu);
    return true;
}

