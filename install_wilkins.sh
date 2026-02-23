#!/bin/bash

# Wilkins Installation Script for ndarray
# This script installs Henson, LowFive, and Wilkins using your specific installations

set -e  # Exit on error

# Configuration - using your specific installations
export MPI_ROOT=/Users/guo.2154/local/mpich-5.0.0
export HDF5_ROOT=$HOME/local/hdf5-1.14.6-mpich-5.0.0
export PYTHON_ROOT=/Users/guo.2154/local/Python-3.10.6-openssl-1.1.1q
export CMAKE_ROOT=/Users/guo.2154/local/cmake-4.2.3
export INSTALL_PREFIX=$HOME/local

# Tools
export CMAKE=$CMAKE_ROOT/bin/cmake
export MPICXX=$MPI_ROOT/bin/mpicxx
export MPICC=$MPI_ROOT/bin/mpicc
export PYTHON=$PYTHON_ROOT/bin/python3

# Installation directories
export HENSON_INSTALL=$INSTALL_PREFIX/henson
export LOWFIVE_INSTALL=$INSTALL_PREFIX/lowfive
export WILKINS_INSTALL=$INSTALL_PREFIX/wilkins

# Working directory for builds
WORK_DIR=$(pwd)/wilkins_build
mkdir -p $WORK_DIR

echo "========================================="
echo "Wilkins Installation Script"
echo "========================================="
echo ""
echo "Configuration:"
echo "  MPI:     $MPI_ROOT"
echo "  HDF5:    $HDF5_ROOT"
echo "  Python:  $PYTHON_ROOT"
echo "  CMake:   $CMAKE_ROOT"
echo "  Install: $INSTALL_PREFIX"
echo ""
echo "Will install:"
echo "  1. Henson  -> $HENSON_INSTALL"
echo "  2. LowFive -> $LOWFIVE_INSTALL"
echo "  3. Wilkins -> $WILKINS_INSTALL"
echo ""

# Check for non-interactive mode
if [ -z "$WILKINS_AUTO_INSTALL" ]; then
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
else
    echo "Running in non-interactive mode..."
fi

# Step 1: Install Henson
echo ""
echo "========================================="
echo "Step 1: Installing Henson"
echo "========================================="
cd $WORK_DIR

if [ ! -d "henson" ]; then
    echo "Cloning Henson..."
    git clone https://github.com/henson-insitu/henson.git
fi

cd henson
rm -rf build
mkdir -p build && cd build

echo "Configuring Henson..."
$CMAKE .. \
  -DCMAKE_INSTALL_PREFIX=$HENSON_INSTALL \
  -DCMAKE_CXX_COMPILER=$MPICXX \
  -DCMAKE_C_COMPILER=$MPICC \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYTHON_EXECUTABLE=$PYTHON \
  -Wno-dev

echo "Building Henson..."
make -j8

echo "Installing Henson..."
make install || {
    echo "Warning: Some Henson components failed to install (likely Python bindings)."
    echo "Checking if core libraries are installed..."
    if [ -f "$HENSON_INSTALL/lib/libhenson.a" ] && [ -f "$HENSON_INSTALL/lib/libhenson-pmpi-static.a" ]; then
        echo "✓ Core Henson libraries installed successfully"
    else
        echo "Error: Core Henson libraries not found"
        exit 1
    fi
}

echo "✓ Henson installed to $HENSON_INSTALL"

# Step 2: Install LowFive
echo ""
echo "========================================="
echo "Step 2: Installing LowFive"
echo "========================================="
cd $WORK_DIR

if [ ! -d "LowFive" ]; then
    echo "Cloning LowFive..."
    git clone https://github.com/diatomic/LowFive.git
fi

cd LowFive
rm -rf build
mkdir -p build && cd build

echo "Configuring LowFive..."
$CMAKE .. \
  -DCMAKE_INSTALL_PREFIX=$LOWFIVE_INSTALL \
  -DCMAKE_CXX_COMPILER=$MPICXX \
  -DCMAKE_C_COMPILER=$MPICC \
  -DHDF5_ROOT=$HDF5_ROOT

echo "Building LowFive..."
make -j8

echo "Installing LowFive..."
make install

echo "✓ LowFive installed to $LOWFIVE_INSTALL"

# Step 3: Install Wilkins
echo ""
echo "========================================="
echo "Step 3: Installing Wilkins"
echo "========================================="
cd $WORK_DIR

if [ ! -d "wilkins" ]; then
    echo "Cloning Wilkins..."
    git clone https://github.com/orcunyildiz/wilkins.git
fi

cd wilkins
rm -rf build
mkdir -p build && cd build

echo "Configuring Wilkins..."
$CMAKE .. \
  -DCMAKE_CXX_COMPILER=$MPICXX \
  -DCMAKE_C_COMPILER=$MPICC \
  -DCMAKE_INSTALL_PREFIX=$WILKINS_INSTALL \
  -DHENSON_INCLUDE_DIR=$HENSON_INSTALL/include \
  -DHENSON_LIBRARY=$HENSON_INSTALL/lib/libhenson.a \
  -DHENSON_PMPI_LIBRARY=$HENSON_INSTALL/lib/libhenson-pmpi-static.a \
  -DPYTHON_EXECUTABLE=$PYTHON \
  -DLOWFIVE_INCLUDE_DIR=$LOWFIVE_INSTALL/include \
  -DLOWFIVE_LIBRARY=$LOWFIVE_INSTALL/lib/liblowfive.dylib \
  -DLOWFIVE_DIST_LIBRARY=$LOWFIVE_INSTALL/lib/liblowfive-dist.a \
  -DHDF5_ROOT=$HDF5_ROOT

echo "Building Wilkins..."
make -j8

echo "Installing Wilkins..."
make install

echo "✓ Wilkins installed to $WILKINS_INSTALL"

# Summary
echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Add these to your ~/.bashrc or ~/.zshrc:"
echo ""
echo "# Wilkins environment"
echo "export PATH=$MPI_ROOT/bin:\$PATH"
echo "export PATH=$HDF5_ROOT/bin:\$PATH"
echo "export PATH=$PYTHON_ROOT/bin:\$PATH"
echo "export PATH=$WILKINS_INSTALL/bin:\$PATH"
echo ""
echo "export LD_LIBRARY_PATH=$MPI_ROOT/lib:\$LD_LIBRARY_PATH"
echo "export LD_LIBRARY_PATH=$HDF5_ROOT/lib:\$LD_LIBRARY_PATH"
echo ""
echo "export HENSON=$HENSON_INSTALL"
echo "export HDF5_ROOT=$HDF5_ROOT"
echo ""
echo "# For running Wilkins workflows"
echo "export HDF5_VOL_CONNECTOR=\"lowfive under_vol=0;under_info={};\""
echo "export HDF5_PLUGIN_PATH=$LOWFIVE_INSTALL/lib"
echo ""
echo "Next steps:"
echo "1. Source your updated shell config"
echo "2. Build ndarray with Henson support (see docs/WILKINS_SETUP.md section 4)"
echo ""
