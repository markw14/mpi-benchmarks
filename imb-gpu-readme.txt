This git branch is an implementation of IMB-GPU benchmark suite. It includes 
the becnhmarks:

gpu_pt2pt  - pairwise ping-pong style benchmark with MPI_Send()/MPI_Recv()
             sequence which uses GPU memory
gpu_ipt2pt - non-blocking version of gpu_pt2pt with
             MPI_Isend()/MPI_Irecv()/MPI_Wait() and optional (switchable)
             background GPU workload
gpu_allreduce  - collective operation MPI_Allreduce() benchmark on GPU memory.

Build instructions.
-------------------

It is necessary to build dependencies for the benchmark first. This is
automated by script ./download-and-build.sh in src_cpp/GPU/thirdparty
directory.

When build is complete, the benchmark can be built by running 'make' from 
src_cpp/ directory. It is normally required to set the desired MPI C++ compiler
name in CXX environment variable before running make. CXXFLAGS and LDFLAGS
environment variables are also available to be set up before running make.


Benchmark features
------------------

The benchmark suite features several ways to set up GPU device locality for each 
MPI rank and two forms of handling GPU memory: i) the form with explicit copies 
performed directly in benchmark code and ii) the form relying on CUDA-aware direct 
transfers done by MPI library internally. In later case the device pointers are 
passed to MPI API.

Device locality for each rank is set before MPI_Init() call, so it can't be
set on per-rank basis in a portable way. The locality then can be set by
CPU affinity introspection done by benchmark iteself.

'generic' option determines current CPU core affinity of a process and total
number of cores, then chooses the GPU id to work with just comparing 
core-number/number-of-cores ratio to number of visible GPU devices
(CUDA_VISIBLE_DEVICES affects this). No real hardware locality is checked.

'coremap' also determines CPU affinity at runtime and uses the user-given map
to pick the device to use wth respect to hardware locality. It's up to user
how to construct the map, nut some utility may automate this.

'hwloc' uses hwloc-1.11 library at runtime to determine the best hardware locality 
at runtime. It relies on PCI topology info. The mode is experimental and may
fail to find correct PCI bridge or other hardware traits on some systems. If
it fails, the way out is to go back to 'generic' or 'coremap' options.

If 'generic' or 'coremap' options see that there is no CPU affinity set,
they refuse to pick a GPU device, and CUDA library will run with default
device as it is described in CUDA docs.

Please note, that 'hwloc' and 'coremap' options ignore CUDA_VISIBLE_DEVICES
setting.

Optional background device workload procedure is implemented for gpu_ipt2pt
benchmark. If switched on, benchmark code submits asynchronous caculation 
workload, followed by optional device-to-host memcopy procedure, which is 
asynchronous as well. The time of workload execution and amount of data to
transfer can be set up with command line options. If workload execution time 
is big enough, it will overlap all the MPI communications procedures.
Benchmark checks on each iteration if provious asynchronous execution is
finished already and starts a new one.



Most important options
-----------------------

-len INT,INT,...  - which message lengths to test (number of elements of a specific MPI datatype, not bytes)
-datatype double|float|int|char  - which MPI datatype to test
-ncycles  - number of test cycles
-nwarmup  - number of warm-up cycles
-mode naive|cudaaware  - MPI/CUDA interaction mode: 'naive' means that data is explicitely copied by benchmark code 
                         to and from device memory before and after relevant MPI calls. 'cudaaware' means that
                         MPI fuctions take device pointers as arguments and do apropriate device-host
                         transfers on their own. 'cudaaware' mode requires support of this 'extension' from MPI 
                         library, otherwise benchmark in this mode may just
                         crash with memory error.
-gpuselect coremap|hwloc|generic  - the way benchmark sets up the GPU device id for each rank. 'coremap' means that
                                    there is a mapping between cores and GPU-ids, which is given explicitely with 
                                    'coretogpu' option. The mapping string in the expected form can be received 
                                    in automated way with some third-party utilities. 'hwloc' means that nearest 
                                    GPU for each rank is deduced by hwloc library calls (see hwloc_iface.cpp). hwloc
                                    library source code is supplied in third-party directory. 'generic' is the way
                                    when number of currently visible CUDA devices is spread among ranks on a node in
                                    a fair way regardless the hardware aspects. The order of ranks in a node is 
                                    determined indirectly by checking the affinity mask at runtime. If no affinity
                                    is set, 'generic' mode doesn't do any active device setup.
-coretogpu <MAP>  - sets up a core to GPU map. The form is: coreN,coreM,...@gpuX;coreP,coreQ,...@gpuY;...
                    where <coreN> is a non-negative integer number meaning the physical core number, <gpuN> is 
                    non-negative integer number meaning the GPU device id as it is perceived by CUDA runtime. Core 
                    groups corresponding to certain GPU device are separated by a semicolon. 
                    NOTE: all physical cores must be enumerated in the list! otherwise program will fail with assert.
                    There is a more compact form of the same list: <HEXBITMASK1>@gpuX;<HEXBITMASK2>@gpuY;...
                    where the cores list: coreN,coreM,... is just represented by bits in a bitmask 
                    (like: 1,3,5 -> bitmask 101010 -> 0x2a)
                    Bitmask must start with '0x' symbols.
-workload INT[,INT]  - sets up the parameters of optionsl background device
                       workload procedure. First integer sets up the number of
                       internal calculation cycles for each workload portion. Single cycle is tuned
                       at runtime to be approximately 1 millisecond long. If
                       If the number of cycles is 0 (the default value), no worload procedure is
                       used. Second integer value for this option is amount of data in bytes to do
                       background memory transfer after each workload portion finish. If this value
                       is 0, no background memory transfers performed.

  

