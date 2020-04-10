#!/bin/sh
#-*-sh-*-

#
# Copyright © 2009 CNRS
# Copyright © 2009-2017 Inria.  All rights reserved.
# Copyright © 2009 Université Bordeaux
# See COPYING in top-level directory.
#

HWLOC_top_builddir="/s/ls4/users/a_medvedev/src/imb-gpu/src_cpp/GPU/thirdparty/hwloc-1.11.13"
distrib="$HWLOC_top_builddir/utils/hwloc/hwloc-distrib"
HWLOC_top_srcdir="/s/ls4/users/a_medvedev/src/imb-gpu/src_cpp/GPU/thirdparty/hwloc-1.11.13"

HWLOC_PLUGINS_PATH=${HWLOC_top_builddir}/src
export HWLOC_PLUGINS_PATH

: ${TMPDIR=/tmp}
{
  tmp=`
    (umask 077 && mktemp -d "$TMPDIR/fooXXXXXX") 2>/dev/null
  ` &&
  test -n "$tmp" && test -d "$tmp"
} || {
  tmp=$TMPDIR/foo$$-$RANDOM
  (umask 077 && mkdir "$tmp")
} || exit $?
file="$tmp/test-hwloc-distrib.output"

set -e
(
  $distrib --if synthetic --input "2 2 2" 2
  echo
  $distrib --if synthetic --input "2 2 2" 4
  echo
  $distrib --if synthetic --input "2 2 2" 8
  echo
  $distrib --if synthetic --input "2 2 2" 13
  echo
  $distrib --if synthetic --input "2 2 2" 16
  echo
  $distrib --if synthetic --input "3 3 3" 4
  echo
  $distrib --if synthetic --input "3 3 3" 4 --single
  echo
  $distrib --if synthetic --input "3 3 3" 4 --reverse
  echo
  $distrib --if synthetic --input "3 3 3" 4 --reverse --single
  echo
  $distrib --if synthetic --input "4 4" 2
  echo
  $distrib --if synthetic --input "4 4" 2 --single
  echo
  $distrib --if synthetic --input "4 4" 2 --reverse --single
  echo
  $distrib --if synthetic --input "4 4 4 4" 19
  echo

  $distrib --if synthetic --input "2 2 2 2" 9
  echo
  $distrib --if synthetic --input "2 2 2 2" --from pu 9
  echo
  $distrib --if synthetic --input "2 2 2 2" --to core 9
  echo
) > "$file"
diff -u $HWLOC_top_srcdir/utils/hwloc/test-hwloc-distrib.output "$file"
rm -rf "$tmp"
