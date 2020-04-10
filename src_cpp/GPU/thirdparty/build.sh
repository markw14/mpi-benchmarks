set -e 


cd yaml-cpp-*
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

cd argsparser && make && cd ..

cp -v argsparser/argsparser.h include/
cp -v argsparser/libargsparser.so lib/
cp -av yaml-cpp/include/yaml-cpp include/ 
cp -av yaml-cpp/lib/* lib/
cp -rv hwloc/include/* include/
cp -rv hwloc/lib/* lib/
 
