
cd /tmp
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.10.0  # or a specific version
cd ../opencv_contrib
git checkout 4.10.0
cd ../opencv
mkdir build && cd build
unset OPAL_PREFIX
unset OPENMPI_VERSION
unset OMPI_MCA_coll_hcoll_enable
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/usr/local/mpi/bin' | paste -sd ':' -)
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_MPI=OFF \
      -D WITH_CUBLAS=1 \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF ..
make -j$(nproc)
sudo make install
sudo ldconfig