/*****************************************************************************
 *                                                                           *
 * Copyright 2016-2018 Intel Corporation.                                    *
 *                                                                           *
 *****************************************************************************

This code is covered by the Community Source License (CPL), version
1.0 as published by IBM and reproduced in the file "license.txt" in the
"license" subdirectory. Redistribution in source and binary form, with
or without modification, is permitted ONLY within the regulations
contained in above mentioned license.

Use of the name and trademark "Intel(R) MPI Benchmarks" is allowed ONLY
within the regulations of the "License for Use of "Intel(R) MPI
Benchmarks" Name and Trademark" as reproduced in the file
"use-of-trademark-license.txt" in the "license" subdirectory.

THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT
LIMITATION, ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT,
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. Each Recipient is
solely responsible for determining the appropriateness of using and
distributing the Program and assumes all risks associated with its
exercise of rights under this Agreement, including but not limited to
the risks and costs of program errors, compliance with applicable
laws, damage to or loss of data, programs or equipment, and
unavailability or interruption of operations.

EXCEPT AS EXPRESSLY SET FORTH IN THIS AGREEMENT, NEITHER RECIPIENT NOR
ANY CONTRIBUTORS SHALL HAVE ANY LIABILITY FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING
WITHOUT LIMITATION LOST PROFITS), HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OR
DISTRIBUTION OF THE PROGRAM OR THE EXERCISE OF ANY RIGHTS GRANTED
HEREUNDER, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

EXPORT LAWS: THIS LICENSE ADDS NO RESTRICTIONS TO THE EXPORT LAWS OF
YOUR JURISDICTION. It is licensee's responsibility to comply with any
export regulations applicable in licensee's jurisdiction. Under
CURRENT U.S. export regulations this software is eligible for export
from the U.S. and can be downloaded by or otherwise exported or
reexported worldwide EXCEPT to U.S. embargoed destinations which
include Cuba, Iraq, Libya, North Korea, Iran, Syria, Sudan,
Afghanistan and any other country to which the U.S. has embargoed
goods and services.

 ***************************************************************************
*/

#include "gpu_benchmark.h"
#include "yaml_io.h"
#include "alloc.h"
#include "device_routines.h"
#include <unistd.h>

namespace gpu_suite {

    inline bool set_stride(int rank, int size, int &stride, int &group)
    {
        if (stride == 0)
            stride = size / 2;
        if (stride <= 0 || stride > size / 2)
            return false;
        group = rank / stride;
        if ((group / 2 == size / (2 * stride)) && (size % (2 * stride) != 0))
            return false;
        return true;
    }


    void GPUBenchmark::init() {
        GET_PARAMETER(std::vector<int>, len);
        GET_PARAMETER(MPI_Datatype, datatype);
        GET_PARAMETER(std::string, mode);
        scope = std::make_shared<VarLenScope>(len);
        MPI_Type_size(datatype, &dtsize);
        size_t size_to_alloc = (size_t)scope->get_max_len() * (size_t)dtsize;
        if (size_to_alloc <= ASSUMED_CACHE_SIZE * 3)
            size_to_alloc = ASSUMED_CACHE_SIZE * 3;
        device_mem_alloc(device_sbuf, size_to_alloc);
        device_mem_alloc(device_rbuf, size_to_alloc);
        host_mem_alloc(host_sbuf, size_to_alloc);
        host_mem_alloc(host_rbuf, size_to_alloc);
        if (host_rbuf == nullptr || host_sbuf == nullptr)
            throw std::runtime_error("GPUBenchmark: memory allocation error.");
        if (device_rbuf == nullptr || device_sbuf == nullptr)
            throw std::runtime_error("GPUBenchmark: memory allocation error.");
        allocated_size = size_to_alloc;
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (mode == "cudaaware") {
            is_cuda_aware = true;
        }
    }
    
    void GPUBenchmark::run(const scope_item &item) { 
        GET_PARAMETER(MPI_Datatype, datatype);
        GET_PARAMETER(int, ncycles);
        GET_PARAMETER(int, nwarmup);
        double time; 
        bool done = benchmark(item.len, datatype, nwarmup, ncycles, time);
        if (!done) {
            results[item.len] = result { false, 0.0 };
        }
    }

    void GPUBenchmark::finalize() { 
#ifdef WITH_YAML_CPP        
        GET_PARAMETER(YAML::Emitter, yaml_out);
        YamlOutputMaker yaml_tmin("tmin");
        YamlOutputMaker yaml_tmax("tmax");
        YamlOutputMaker yaml_tavg("tavg");
        YamlOutputMaker yaml_topo("topo");
#endif        
        for (auto it = results.begin(); it != results.end(); ++it) {
            int len = it->first;
            double time = (it->second).time, tmin = 0, tmax = 0, tavg = 0;
            int is_done = ((it->second).done ? 1 : 0), nexec = 0;
            MPI_Reduce(&is_done, &nexec, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!(it->second).done) time = 1e32;
            MPI_Reduce(&time, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            if (!(it->second).done) time = 0.0;
            MPI_Reduce(&time, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&time, &tavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                if (nexec == 0) {
                    std::cout << get_name() << ": " << "{ " 
                              << "len: " << len << ", "
                              << " error: \"no successful executions, check params!\"" 
                              << " }" << std::endl;
                } else {
                    tavg /= nexec;
                    std::cout << get_name() << ": " << "{ " << "len: " << len << ", "
                        << " time: [ " << tmin << ", " 
                                      << tavg << ", " 
                                      << tmax << " ]" 
                                      << " }" << std::endl;
#ifdef WITH_YAML_CPP                    
                    yaml_tmin.add(len, tmin);
                    yaml_tmax.add(len, tmax);
                    yaml_tavg.add(len, tavg);
#endif                    
                }
            }
        }
#ifdef WITH_YAML_CPP        
        yaml_topo.add("np", np);
        WriteOutYaml(yaml_out, get_name(), {yaml_tavg, yaml_topo});
#endif
        // NOTE: can't free pinned memory in destructor, CUDA runtime complains 
        // it's too late
        host_mem_free(host_sbuf);
        host_mem_free(host_rbuf);
        device_mem_free(device_sbuf);
        device_mem_free(device_rbuf);
    }

    char *GPUBenchmark::get_sbuf() {
        if (!is_cuda_aware) {
            return host_sbuf;
        } else {
            return device_sbuf;
        }
    }

    char *GPUBenchmark::get_rbuf() {
        if (!is_cuda_aware) {
            return host_rbuf;
        } else {
            return device_rbuf;
        }
    }

    void GPUBenchmark::update_sbuf(char *b, size_t off, size_t size) {
        if (b == host_sbuf) {
            d2h_transfer(b + off, device_sbuf, size);
        }        
    }

    void GPUBenchmark::update_rbuf(char *b, size_t off, size_t size) {
        if (b == host_rbuf) {
            h2d_transfer(device_rbuf, b + off, size);
        }        
    }

    bool GPUBenchmark_pt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, 
                                       int ncycles, double &time) {
        int stride = 0, group;
        if (!set_stride(rank, np, stride, group)) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = (size_t)count * (size_t)dtsize;
        size_t n = allocated_size / b;
        double t1 = 0, t2 = 0;
        const int tag = 1;
        int pair = -1;
        if (group % 2 == 0) {
            pair = rank + stride;
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                update_sbuf(get_sbuf(), (i%n)*b, b);
                MPI_Send((char*)get_sbuf() + (i%n)*b, count, datatype, pair, 
                          tag, MPI_COMM_WORLD);
                MPI_Recv((char*)get_rbuf() + (i%n)*b, count, datatype, pair, 
                          MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                update_rbuf(get_rbuf(), (i%n)*b, b);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
        } else {
            pair = rank - stride;
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                MPI_Recv((char*)get_rbuf() + (i%n)*b, count, datatype, pair, 
                         MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                update_rbuf(get_rbuf(), (i%n)*b, b);
                update_sbuf(get_sbuf(), (i%n)*b, b);
                MPI_Send((char*)get_sbuf() + (i%n)*b, count, datatype, pair, 
                         tag, MPI_COMM_WORLD);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
        } 
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time };
        return true;
    }

    void GPUBenchmark_ipt2pt::init() {
        GPUBenchmark::init();
        calc.init();
    }

    bool GPUBenchmark_ipt2pt::benchmark(int count, MPI_Datatype datatype, int nwarmup, 
                                        int ncycles, double &time) {
        int stride = 0, group;
        if (!set_stride(rank, np, stride, group)) {
            MPI_Barrier(MPI_COMM_WORLD);
            return false;
        }
        size_t b = (size_t)count * (size_t)dtsize;
        size_t n = allocated_size / b;
        double t1 = 0, t2 = 0, total_ctime = 0, local_ctime = 0;
        const int tag = 1;
        int pair = -1;
        MPI_Request request[2];
        if (group % 2 == 0) {
            pair = rank + stride;
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                update_sbuf(get_sbuf(), (i%n)*b, b);
                MPI_Isend((char*)get_sbuf()  + (i%n)*b, count, datatype, pair, 
                          tag, MPI_COMM_WORLD, &request[0]);
                MPI_Irecv((char*)get_rbuf() + (i%n)*b, count, datatype, pair, 
                          MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
                calc.benchmark(count, datatype, 0, 1, local_ctime);
                if (i >= nwarmup) {
                    total_ctime += local_ctime;
                }
                MPI_Waitall(2, request, MPI_STATUSES_IGNORE);
                update_rbuf(get_rbuf(), (i%n)*b, b);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
        } else {
            pair = rank - stride;
            for (int i = 0; i < ncycles + nwarmup; i++) {
                if (i == nwarmup) t1 = MPI_Wtime();
                update_sbuf(get_sbuf(), (i%n)*b, b);
                MPI_Isend((char*)get_sbuf() + (i%n)*b, count, datatype, pair, 
                          tag, MPI_COMM_WORLD, &request[0]);
                MPI_Irecv((char*)get_rbuf() + (i%n)*b, count, datatype, pair, 
                          MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
                calc.benchmark(count, datatype, 0, 1, local_ctime);
                if (i >= nwarmup) {
                    total_ctime += local_ctime;
                }
                MPI_Waitall(2, request, MPI_STATUSES_IGNORE);
                update_rbuf(get_rbuf(), (i%n)*b, b);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
        } 
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time };
        device_sync_context();
        return true;
    }

    void barrier(int rank, int np) {
#if 0
        (void)rank;
        (void)np;
        MPI_Barrier(MPI_COMM_WORLD);
#else

        int mask = 0x1;
        int dst, src;
        int tmp = 0;
        for (; mask < np; mask <<= 1) {
            dst = (rank + mask) % np;
            src = (rank - mask + np) % np;
            MPI_Sendrecv(&tmp, 0, MPI_BYTE, dst, 1010,
                         &tmp, 0, MPI_BYTE, src, 1010,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
#endif
    }

    bool GPUBenchmark_allreduce::benchmark(int count, MPI_Datatype datatype, int nwarmup, 
                                           int ncycles, double &time) {
        size_t b = (size_t)count * (size_t)dtsize;
        size_t n = allocated_size / b;
        double t1 = 0, t2 = 0;
        time = 0;
        for (int i = 0; i < ncycles + nwarmup; i++) {
            if (i >= nwarmup) t1 = MPI_Wtime();
            update_sbuf(get_sbuf(), (i%n)*b, b);
            MPI_Allreduce((char *)get_sbuf() + (i%n)*b, (char *)get_rbuf() + (i%n)*b, count, 
                          datatype, MPI_SUM, MPI_COMM_WORLD);
            update_rbuf(get_rbuf(), (i%n)*b, b);
            if (i >= nwarmup) {
                t2 = MPI_Wtime();
                time += (t2 - t1);
            }
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
            barrier(rank, np);
        }
        time /= ncycles;
        MPI_Barrier(MPI_COMM_WORLD);
        results[count] = result { true, time };
        return true;
    }

    void GPUBenchmark_calc::init() {
        GPUBenchmark::init();
        GET_PARAMETER(int, workload_cycles);
        GET_PARAMETER(int, workload_transfer_size);
        if (workload_cycles && workload_transfer_size) {
            device_mem_alloc(device_transf_buf, workload_transfer_size);
            host_mem_alloc(host_transf_buf, workload_transfer_size);
        }

        // Workload execution time calibration procedure. Trying to tune number of cycles 
        // so that workload execution+sync time is about 1000 usec
        workload_calibration = 1;
        for (int i = 0; i < 10; i++) {             
            timer t;
            device_submit_workload(1, workload_calibration);
            device_sync_context();
            int usec = t.stop();
            if (i < 3)
                continue;
            if (usec == 0)
                break;
            if (usec < 900L) {
                workload_calibration += (1 + int(1000L/usec));
            }
        }
    }

    bool GPUBenchmark_calc::benchmark(int count, MPI_Datatype datatype, int nwarmup, int ncycles, 
                                      double &time) {
        (void)count;
        (void)datatype;
        (void)nwarmup;
        (void)ncycles;
        (void)time;
        GET_PARAMETER(int, workload_cycles);
        GET_PARAMETER(int, workload_transfer_size);
        if (!workload_cycles)
            return true;
        if (device_is_idle()) {
            device_submit_workload(workload_cycles, workload_calibration);
            if (workload_transfer_size) {
                d2h_transfer(host_transf_buf, device_transf_buf, workload_transfer_size, 
                             transfer_t::WORKLOAD);
            }
        }
        return true;
    }
    
    void GPUBenchmark_calc::finalize() {
        GPUBenchmark::finalize();
        GET_PARAMETER(int, workload_cycles);
        GET_PARAMETER(int, workload_transfer_size);
        if (workload_cycles && workload_transfer_size) {
            device_mem_free(device_transf_buf);
            host_mem_free(host_transf_buf);
        }
    }

    DECLARE_INHERITED(GPUBenchmark_pt2pt, gpu_pt2pt)
    DECLARE_INHERITED(GPUBenchmark_ipt2pt, gpu_ipt2pt)
    DECLARE_INHERITED(GPUBenchmark_allreduce, gpu_allreduce)
    DECLARE_INHERITED(GPUBenchmark_calc, gpu_calc)
}
