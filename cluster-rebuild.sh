module load cmake/3.23.2
cd build
cmake --build . --config RelWithDebInfo --target tungsten -j 4
cd ..
