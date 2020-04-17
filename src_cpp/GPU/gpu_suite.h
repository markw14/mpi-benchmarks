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

#include <mpi.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>

#include "benchmark.h"
#include "benchmark_suites_collection.h"
#include "scope.h"
#include "utils.h"
#include "args_parser.h"
#include "alloc.h"

extern bool gpu_conf_init(const std::string &);
#ifdef WITH_HWLOC
extern bool gpu_conf_init_with_hwloc();
#endif

namespace gpu_suite {

    #include "benchmark_suite.h"

    DECLARE_BENCHMARK_SUITE_STUFF(BS_GENERIC, gpu_suite)

    template <> bool BenchmarkSuite<BS_GENERIC>::declare_args(args_parser &parser,
                                                              std::ostream &output) const {
        UNUSED(output);
        parser.set_current_group(get_name());
        parser.add_vector<int>("len", "4,128,2048,32768,524288").
                     set_mode(args_parser::option::APPLY_DEFAULTS_ONLY_WHEN_MISSING).
                     set_caption("INT,INT,...");
        parser.add<std::string>("datatype", "double").
                     set_caption("double|float|int|char");
        parser.add<int>("stride", 0);
        parser.add<int>("ncycles", 1000);
        parser.add<int>("nwarmup", 3); 
        parser.add<std::string>("mode", "naive").
                     set_caption("naive|cudaaware");
        parser.add<std::string>("allocmode", "mpi").
                     set_caption("mpi|cuda");
#ifdef WITH_HWLOC        
        parser.add<std::string>("gpuselect", "generic").
                     set_caption("coremap|hwloc|generic");
#else
        parser.add<std::string>("gpuselect", "generic").
                     set_caption("coremap|generic");
#endif        
        parser.add<std::string>("coretogpu", "").
                     set_caption("- core to GPU devices map, like: 0,1,2,3@0;4,5,6,7@1");
        parser.add_vector<int>("workload", "0,0", ',', 1, 2).
                     set_caption("- background workload tuning: calc_cycles[,transfer_size]");
        parser.set_default_current_group();
        return true;
    }

    std::vector<int> len;
    MPI_Datatype datatype;
#ifdef WITH_YAML_CPP        
    YAML::Emitter yaml_out;
#endif    
    std::string yaml_outfile;
    int stride;
    int ncycles, nwarmup;
    host_alloc_t atype;
    std::string mode;
    int workload_cycles, workload_transfer_size;
    template <> bool BenchmarkSuite<BS_GENERIC>::prepare(const args_parser &parser,
                                                         const std::vector<std::string> &,
                                                         const std::vector<std::string> &unknown_args,
                                                         std::ostream &output) {
        if (unknown_args.size() != 0) {
            output << "Some unknown options or extra arguments. Use -help for help." << std::endl;
            return false;
        }
        parser.get<int>("len", len);
        std::string dt = parser.get<std::string>("datatype");
        if (dt == "int") datatype = MPI_INT;
        else if (dt == "double") datatype = MPI_DOUBLE;
        else if (dt == "float") datatype = MPI_FLOAT;
        else if (dt == "char") datatype = MPI_CHAR;
        else {
            output << get_name() << ": " << "Unknown data type in 'datatype' option."
                                            " Use -help for help." << std::endl;
            return false;
        }
        stride = parser.get<int>("stride");
        ncycles = parser.get<int>("ncycles");
        nwarmup = parser.get<int>("nwarmup");
#ifdef WITH_YAML_CPP        
        yaml_outfile = parser.get<std::string>("output");
        yaml_out << YAML::BeginDoc;
        yaml_out << YAML::BeginMap;
#endif
        mode = parser.get<std::string>("mode");
        if (mode != "naive" && mode != "cudaaware") {
            output << get_name() << ": " << "Unknown device interaction mode in 'mode' option."
                                            " Use -help for help." << std::endl;
            return false;
        }
/*
 * NOTE: this is correct only for OpenMPI        
        if (mode == "cudaaware") {
#if !defined(MPIX_CUDA_AWARE_SUPPORT) || !MPIX_CUDA_AWARE_SUPPORT
            output << get_name() << ": " << "CUDA-aware device interaction mode"
                                            " is not supported" << std::endl;
            return false;
#endif            
        }
*/        
        std::string atype_str = parser.get<std::string>("allocmode");
        if (atype_str == "mpi") {
            atype = host_alloc_t::ALLOC_MPI;
        } else if (atype_str == "cuda") {
            atype = host_alloc_t::ALLOC_CUDA;
        } else {
            output << get_name() << ": " << "Unknown host memory allocation mode in 'allocmode' option."
                                            " Use -help for help." << std::endl;
            return false;
        }
        std::string coremapstr;
        std::string gpuselect = parser.get<std::string>("gpuselect");
        if (gpuselect == "coremap") {
            
            if (parser.get<std::string>("coretogpu") == "") {
                output << get_name() << ": " << "'coremap' device selection mode implies"
                                                " the option 'coretogpu' to be set" << std::endl;
                return false;
            }
            coremapstr = parser.get<std::string>("coretogpu");
        } 
#ifndef WITH_HWLOC
        else if (gpuselect == "hwloc") {
            output << get_name() << ": " << "Can't use 'hwloc' device selection mode:"
                                            " built without hwloc support" << std::endl;
            return false;
#endif            
        } else if (gpuselect == "generic") {
            coremapstr = ""; // for generic mode of devices selection
        } else {
            output << get_name() << ": " << "Unknown device selection mode in 'gpuselect' option."
                                            " Use -help for help." << std::endl;
            return false;
        }
        if (parser.get<std::string>("coretogpu") != "" && gpuselect != "coremap") {
            output << get_name() << ": " << "coretogpu option is implies 'coremap'"
                                            " device selection mode to be set." << std::endl; 
        }
        std::vector<int> wrld_opts;
        parser.get<int>("workload", wrld_opts);
        workload_cycles = wrld_opts[0];
        workload_transfer_size = wrld_opts[1];
        
#ifdef WITH_HWLOC
        if (gpuselect == "hwloc") {
            if (!gpu_conf_init_with_hwloc())
                return false;     
        } else {
            if (!gpu_conf_init(coremapstr)) {
                return false;
            }
        }
#else        
        if (!gpu_conf_init(coremapstr))
            return false;     
#endif        
        return true;
    }

     template <> void BenchmarkSuite<BS_GENERIC>::finalize(const std::vector<std::string> &,
                          std::ostream &) {
#ifdef WITH_YAML_CPP        
        yaml_out << YAML::EndMap;
        yaml_out << YAML::Newline;
        if (!yaml_outfile.empty()) {
            std::ofstream ofs(yaml_outfile, std::ios_base::out | std::ios_base::trunc);
            ofs << yaml_out.c_str();
        }
#endif        
    }

#define HANDLE_PARAMETER(TYPE, NAME) if (key == #NAME) { \
                                        result = std::shared_ptr< TYPE >(&NAME, []( TYPE *){}); \
                                     }

#define GET_PARAMETER(TYPE, NAME) TYPE *p_##NAME = suite->get_parameter(#NAME).as< TYPE >(); \
                                  assert(p_##NAME != NULL); \
                                  TYPE &NAME = *p_##NAME;

    template <> any BenchmarkSuite<BS_GENERIC>::get_parameter(const std::string &key) {
        any result;
        HANDLE_PARAMETER(std::vector<int>, len);
        HANDLE_PARAMETER(MPI_Datatype, datatype);
#ifdef WITH_YAML_CPP        
        HANDLE_PARAMETER(YAML::Emitter, yaml_out);
#endif        
        HANDLE_PARAMETER(int, stride);
        HANDLE_PARAMETER(int, ncycles);
        HANDLE_PARAMETER(int, nwarmup);
        HANDLE_PARAMETER(std::string, mode);
        HANDLE_PARAMETER(host_alloc_t, atype);
        HANDLE_PARAMETER(int, workload_cycles);
        HANDLE_PARAMETER(int, workload_transfer_size);
        return result;
    }
}
