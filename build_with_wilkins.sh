#!/bin/bash

# Build ndarray with Wilkins/Henson support
# Run this after installing Henson, LowFive, and Wilkins

set -e  # Exit on error

# Configuration - using your specific installations
export MPI_ROOT=/Users/guo.2154/local/mpich-5.0.0
export HDF5_ROOT=$HOME/local/hdf5-1.14.6-mpich-5.0.0
export CMAKE_ROOT=/Users/guo.2154/local/cmake-4.2.3
export HENSON=$HOME/local/henson

# Tools
export CMAKE=$CMAKE_ROOT/bin/cmake
export MPICXX=$MPI_ROOT/bin/mpicxx
export MPICC=$MPI_ROOT/bin/mpicc

# Build directory
BUILD_DIR=build_wilkins
NDARRAY_ROOT=$(pwd)

echo "========================================="
echo "Building ndarray with Wilkins Support"
echo "========================================="
echo ""
echo "Configuration:"
echo "  MPI:    $MPI_ROOT"
echo "  HDF5:   $HDF5_ROOT"
echo "  CMake:  $CMAKE_ROOT"
echo "  Henson: $HENSON"
echo ""

# Check if Henson is installed
if [ ! -f "$HENSON/lib/libhenson.a" ]; then
    echo "ERROR: Henson not found at $HENSON"
    echo "Please install Henson first using install_wilkins.sh"
    exit 1
fi

# Create build directory
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Configuring ndarray..."
$CMAKE .. \
  -DCMAKE_CXX_COMPILER=$MPICXX \
  -DCMAKE_C_COMPILER=$MPICC \
  -DNDARRAY_USE_HDF5=TRUE \
  -DNDARRAY_USE_HENSON=TRUE \
  -DNDARRAY_USE_MPI=TRUE \
  -DNDARRAY_BUILD_TESTS=ON \
  -DNDARRAY_BUILD_EXAMPLES=ON \
  -DHDF5_ROOT=$HDF5_ROOT \
  -DHENSON=$HENSON

echo ""
echo "Building ndarray..."
make -j8

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
echo "Build directory: $NDARRAY_ROOT/$BUILD_DIR"
echo ""
echo "To test:"
echo "  cd $BUILD_DIR"
echo "  make test"
echo ""
echo "Library location:"
echo "  $NDARRAY_ROOT/$BUILD_DIR/lib/libndarray.so"
echo ""
