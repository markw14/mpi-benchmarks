#!/bin/sh
#-*-sh-*-

#
# Copyright © 2012-2018 Inria.  All rights reserved.
# See COPYING in top-level directory.
#

HWLOC_top_builddir="/s/ls4/users/a_medvedev/src/imb-gpu/src_cpp/GPU/thirdparty/hwloc-1.11.13"
xmlbuffer=xmlbuffer

HWLOC_PLUGINS_PATH=${HWLOC_top_builddir}/src
export HWLOC_PLUGINS_PATH

if test "`basename $1`" = "$xmlbuffer"; then
    "$@" 1 1
    "$@" 0 1
    "$@" 1 0
    "$@" 0 0
else
    "$@"
fi
