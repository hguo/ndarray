#!/bin/bash
# Template for HPC site-specific configuration
# Copy this file and customize for your HPC center

# System information
SITE_NAME="Your HPC Center Name"
SYSTEM_NAME="System Name (e.g., Summit, Cori)"
ARCHITECTURE="CPU/GPU architecture"

echo "========================================"
echo "ndarray build configuration"
echo "Site: ${SITE_NAME}"
echo "System: ${SYSTEM_NAME}"
echo "========================================"

# Load required modules
# Customize these for your system
module purge  # Optional: clean module environment
module load cmake/3.20.0
module load gcc/11.2.0
# module load intel/2021.4.0  # Or Intel compiler
# module load mpi/openmpi/4.1.1  # Or other MPI

# Optional: NetCDF support
# module load netcdf-c/4.8.1
# module load netcdf-cxx/4.3.1

# Optional: HDF5 support
# module load hdf5/1.12.1

# Optional: Parallel I/O
# module load parallel-netcdf/1.12.2

# Optional: ADIOS2 for in-situ
# module load adios2/2.8.0

# Optional: VTK for visualization
# module load vtk/9.1.0

# Optional: YAML for stream configs
# module load yaml-cpp/0.7.0

# List loaded modules
echo ""
echo "Loaded modules:"
module list

# Export CMake flags
export CMAKE_FLAGS=""

# Compiler selection
export CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_CXX_COMPILER=$(which mpicxx)"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_C_COMPILER=$(which mpicc)"

# Enable/disable features based on available modules
export CMAKE_FLAGS="${CMAKE_FLAGS} -DNDARRAY_USE_MPI=AUTO"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DNDARRAY_USE_NETCDF=AUTO"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DNDARRAY_USE_HDF5=AUTO"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DNDARRAY_USE_PNETCDF=AUTO"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DNDARRAY_USE_ADIOS2=AUTO"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DNDARRAY_USE_VTK=AUTO"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DNDARRAY_USE_YAML=AUTO"

# Optional: Set installation prefix
# export CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_INSTALL_PREFIX=$HOME/software/ndarray"

# Optional: Build type
export CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_BUILD_TYPE=Release"

# Optional: Enable tests
export CMAKE_FLAGS="${CMAKE_FLAGS} -DNDARRAY_BUILD_TESTS=ON"

echo ""
echo "CMake flags:"
echo ${CMAKE_FLAGS}
echo ""
echo "To build:"
echo "  mkdir build && cd build"
echo "  cmake .. ${CMAKE_FLAGS}"
echo "  make -j8"
echo ""
