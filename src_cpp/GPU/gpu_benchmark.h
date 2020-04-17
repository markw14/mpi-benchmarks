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

#include "gpu_suite.h"

namespace gpu_suite {

    class GPUBenchmark : public Benchmark {
        public:
        const size_t ASSUMED_CACHE_SIZE = 4 * 1024 * 1024;
        struct result {
            bool done;
            double time;
        };
        std::map<int, result> results;
        char *host_sbuf = nullptr, *host_rbuf = nullptr;
        char *device_sbuf = nullptr, *device_rbuf = nullptr;
        int np, rank;
        size_t allocated_size;
        int dtsize;
        bool is_cuda_aware = false;
        public:
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int stride, 
                               int nwarmup, int ncycles, double &time) = 0;
        virtual void run(const scope_item &item) override; 
        virtual void finalize() override;
        GPUBenchmark() : host_sbuf(nullptr), host_rbuf(nullptr), 
                         device_sbuf(nullptr), device_rbuf(nullptr), np(0), 
                         rank(0), allocated_size(0), dtsize(0) {}
        char *get_sbuf();
        char *get_rbuf();
        void update_sbuf(char *, size_t off, size_t size);
        void update_rbuf(char *, size_t off, size_t size);
        virtual ~GPUBenchmark() {}
    };

    class GPUBenchmark_calc : public GPUBenchmark {
        public:
        char *host_transf_buf = nullptr, *device_transf_buf = nullptr;
        int workload_calibration = 0;
        public:
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int stride,
                               int nwarmup, int ncycles, double &time) override;
        virtual void finalize() override;
        DEFINE_INHERITED(GPUBenchmark_calc, BenchmarkSuite<BS_GENERIC>);
    };

    class GPUBenchmark_pt2pt : public GPUBenchmark {
        public:
        virtual bool benchmark(int count, MPI_Datatype datatype, int stride,
                               int nwarmup, int ncycles, double &time) override;
        virtual void finalize() override { GPUBenchmark::finalize(); };
        DEFINE_INHERITED(GPUBenchmark_pt2pt, BenchmarkSuite<BS_GENERIC>);
    };

    class GPUBenchmark_ipt2pt : public GPUBenchmark {
        public:
        GPUBenchmark_calc calc;
        virtual void init() override;
        virtual bool benchmark(int count, MPI_Datatype datatype, int stride,
                               int nwarmup, int ncycles, double &time) override;
        virtual void finalize() override { GPUBenchmark::finalize(); };
        DEFINE_INHERITED(GPUBenchmark_ipt2pt, BenchmarkSuite<BS_GENERIC>);
    };

    class GPUBenchmark_allreduce : public GPUBenchmark {
        public:
        virtual bool benchmark(int count, MPI_Datatype datatype, int stride,
                               int nwarmup, int ncycles, double &time) override;
        virtual void finalize() override { GPUBenchmark::finalize(); };
        DEFINE_INHERITED(GPUBenchmark_allreduce, BenchmarkSuite<BS_GENERIC>);
    };

}

#include <sys/time.h>

struct timer {
    const time_t sec_to_usec = 1000000L;
    std::string name;
    bool do_out = false;
    bool stopped = false;
    std::stringstream comment;
    timeval tv[2];
    long *presult = nullptr;
    timer(const std::string& _name = "", bool _do_out = false) :
        name(_name), do_out(_do_out) {
            gettimeofday(&tv[0], NULL);
        }
    long time_diff() {
        return ((long)tv[1].tv_sec - (long)tv[0].tv_sec) * sec_to_usec + (long)tv[1].tv_usec - (long)tv[0].tv_usec;
    }
    timer(long *_presult) : presult(_presult) {
        gettimeofday(&tv[0], NULL);
    }
    long stop() {
        gettimeofday(&tv[1], NULL);
        long diff = time_diff();
        if (presult) {
            *presult = diff;
            return diff;
        }
        if (do_out) {
            std::cout << name << ": " << "[ " << "time (usec): " << diff;
            if (comment.str().size()) {
                std::cout << ", " << "comment: \"" << comment.str() << "\"";
            }
            std::cout << " " << "]" << std::endl;
        }
        stopped = true;
        return diff;
    }
    ~timer() throw() {
        if (!stopped)
            stop();
    }
};
