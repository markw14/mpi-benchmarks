#!/bin/bash

[ -f ./env.sh ] && . ./env.sh && env_init_global && env_init IMB-GPU

NNODES=$JOBRUN_OPTS_NNODES
PPN=$JOBRUN_OPTS_PPN
NGPUS=$JOBRUN_OPTS_NGPUS
OPT=./run.options

NNODES=1
PPN=2

function get_rank0() {
   local rank0=$(cat jobrun.out | awk '/^Rank 0 output:/{print $4}')
   [ ! -z "$rank0" ] && echo "$rank0"
}

[ ! -z "$*" ] && ARGS=-a
jobrun.sh -n $NNODES -p $PPN -o $OPT $ARGS "$*" | tee jobrun.out

