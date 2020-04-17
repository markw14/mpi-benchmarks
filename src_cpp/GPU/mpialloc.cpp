#include <mpi.h>
#include "cuda_helpers.h"

void mpi_host_mem_alloc(char *&host_buf, size_t size_to_alloc)
{
    MPI_CALL(MPI_Alloc_mem(size_to_alloc, MPI_INFO_NULL, &host_buf));
}

void mpi_host_mem_free(char *host_buf)
{
    MPI_CALL(MPI_Free_mem(host_buf));
}

