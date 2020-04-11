#!/bin/bash

set -e 


function download() {
    [ ! -f yaml-cpp-0.6.2.tar.gz ] && wget https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz
    [ ! -f v0.0.1.tar.gz ] && wget https://github.com/a-v-medvedev/argsparser/archive/v0.0.1.tar.gz
    [ ! -f hwloc-1.11.13.tar.gz ] && wget https://download.open-mpi.org/release/hwloc/v1.11/hwloc-1.11.13.tar.gz
    true
}

function unpack() {
    [ -e yaml-cpp-0.6.2.tar.gz -a ! -e yaml-cpp-yaml-cpp-0.6.2 ] && tar xzf yaml-cpp-0.6.2.tar.gz
    [ -e v0.0.1.tar.gz -a ! -e argsparser-0.0.1 ] && tar xzf v0.0.1.tar.gz
    [ -e hwloc-1.11.13.tar.gz -a ! -e hwloc-1.11.13 ] && tar xzf hwloc-1.11.13.tar.gz
    cd argsparser-0.0.1 
    [ ! -e yaml-cpp -a ! -L yaml-cpp ] && ln -s ../yaml-cpp yaml-cpp 
    cd ..
}

function build() {
    cd yaml-cpp-yaml-cpp-0.6.2
    [ -e build ] && rm -rf build
    mkdir -p build
    cd build
    cmake -DBUILD_SHARED_LIBS=ON -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_CPP_BUILD_CONTRIB=OFF .. -DCMAKE_INSTALL_PREFIX=$PWD/../../yaml-cpp
    make clean
    make -j8
    make install
    cd ../..

    cd hwloc-1*
    ./configure --prefix=$PWD/../hwloc && make -j8 && make install && cd ..

    cd argsparser-0.0.1 && make && cd ..
}

function install() {
    mkdir -p include
    mkdir -p lib
    cp -v argsparser-0.0.1/argsparser.h include/
    cp -v argsparser-0.0.1/libargsparser.so lib/
    cp -av yaml-cpp/include/yaml-cpp include/ 
    cp -av yaml-cpp/lib/* lib/
    cp -rv hwloc/include/* include/
    cp -rv hwloc/lib/* lib/
}

download
unpack
build
install
