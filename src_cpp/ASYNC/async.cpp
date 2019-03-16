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

#include "async_benchmark.h"

namespace async_suite {

    void AsyncBenchmark::init() {
        GET_PARAMETER(std::vector<int>, len);
        GET_PARAMETER(MPI_Datatype, datatype);
        scope = std::make_shared<VarLenScope>(len);
        int idts;
        MPI_Type_size(datatype, &idts);
        rbuf = (char *)malloc((size_t)scope->get_max_len() * (size_t)idts);
        sbuf = (char *)malloc((size_t)scope->get_max_len() * (size_t)idts);
    }
    void AsyncBenchmark::run(const scope_item &item) { 
        GET_PARAMETER(MPI_Datatype, datatype);
        GET_PARAMETER(int, ncycles);
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (np < 2) {
            std::cout << get_name() << ": two or more ranks required" << std::endl;
            return;
        }
        MPI_Status stat;
        double t1 = 0, t2 = 0, time = 0;
        const int tag = 1;
        if (rank == 0) {
            t1 = MPI_Wtime();
            for(int i = 0; i < ncycles; i++) {
                MPI_Send((char*)sbuf, item.len, datatype, 1, tag, MPI_COMM_WORLD);
                MPI_Recv((char*)rbuf, item.len, datatype, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
        } else if (rank == 1) {
            t1 = MPI_Wtime();
            for(int i = 0; i < ncycles; i++) {
                MPI_Recv((char*)rbuf, item.len, datatype, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
                MPI_Send((char*)sbuf, item.len, datatype, 0, tag, MPI_COMM_WORLD);
            }
            t2 = MPI_Wtime();
            time = (t2 - t1) / ncycles;
        } 
        MPI_Barrier(MPI_COMM_WORLD);
        results[item.len] = time;
    }
    void AsyncBenchmark::finalize() { 
        if (rank == 0) {
            for (std::map<int, double>::iterator it = results.begin();
                    it != results.end(); ++it) {
                std::cout << get_name() << ": " << "len=" << it->first << " time=" << it->second << std::endl; 
            }
        }
    }
    AsyncBenchmark::~AsyncBenchmark() {
        free(rbuf);
        free(sbuf);
    }

    DECLARE_INHERITED(AsyncBenchmark, async)
}
