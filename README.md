# Intel(R) MPI Benchmarks: IMB-ASYNC
[![Common Public License Version 1.0](https://img.shields.io/badge/license-Common%20Public%20License%20Version%201.0-green.svg)](license/license.txt)
![v2019](https://img.shields.io/badge/v.2019-Gold-orange.svg)
--------------------------------------------------

The IMB-ASYNC benchmark suite is a small collection of microbenchmark tools which
help to fairly estimate the MPI message passing asynchronous performance (asynchronity level,
progress performance) in several useful scenarios.

The individual bechmarks include:
- sync_pt2p2, async_pt2pt -- ping-pong style point-to-point benchmark with stride between peers 
given with option "-stride". Synchronous variant utilizes MPI_Send()/MPI_Recv() function calls.
Asynchronous variant uses equivalent MPI_Isend()/MPI_Irev()/MPI_Wait() combination, and pure
calculation workload is optionally called before MPI_Wait() call (see "-workload" option).
- sync_allreduce, async_allreduce -- MPI_Allreduce() and MPI_Iallreduce()/MPI_Wait() benchmarks for the
whole MPI_COMM_WORLD commuicator. Pure calculation workload is optionally called before MPI_Wait() call
(see "-workload" option).
- sync_na2a, async_na2a -- messages exchnage with two closest neighbour ranks for each rank in 
MPI_COMM_WORLD. Implemented with MPI_Neighbor_alltoall() for synchronous variant and with 
MPI_Ineighbor_alltoall()/MPI_Wait() combination for a asynchronous one. Pure calculation workload 
is optionally called before MPI_Wait() call (see "-workload" option).
- sync_rma_pt2pt, async_rma_pt2pt -- ping-pong stype message exchnage with a neighbour rank 
(respectig "-stride" parameter). This is simple a one-sided coccunication version of
sync_pt2pt/async_pt2pt benchmark pair. Implemented with one-sided MPI_Get() call in 
post/start/complete/wait semantics for a synchronous variant and lock/flush/unlock semantics for an 
asynchronous one. Pure calculation workload is optionally called before MPI_Win_unlock() call (see 
"-workload" option).

The "calibration" benchmark:
- async_calibration -- is used to detect and report the calculation cycle calibration constant (later 
used as "-cper10usec" parameter value).

The workload option (meaningful for asynchromous variants of all benchmarks):
- -workload none -- means: do nothing, just proceed with waiting the operation to complete 
with MPI_Wait() or MPI_Win_unlock();
- -workload calc -- spin dummy calculation loop on each rank. The number of calculation cycles
to run is calculated from the constant given in "-calctime" parameter and the "-cper10usec" constant
also given as a parameter. "-cper10usec" parameter value highly depends on the CPU model and speed, 
so it must be obtained beforehand with async_calibration benchmark on each particular machine.
"-cper10usec" parameter is required for "calc" and "calc_progress" workload types.
- -workload calc_progress -- as in "calc" workload type, do some dummy calculation, but also
call MPI_Test() sometimes. "-spinperiod" parameter sets up how often to call MPI_Test().



----------------------
Copyright and Licenses
----------------------

This benchmark suite inherits Common Public License terms of Intel(R) MPI Benchmarks project, 
which it is based on.


(C) Intel Corporation

(C) Alexey V. Medvedev (2019-2020)
