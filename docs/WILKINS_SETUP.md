# Wilkins Integration Guide for ndarray

## Overview

Wilkins is an in-situ workflow system that enables heterogeneous task specification and execution. It uses:
- **LowFive** (HDF5-based library) for data transport
- **Henson** for execution model
- **HDF5 1.14+** for data model

## Your System Configuration

- **HDF5**: ~/local/hdf5-1.14.6-mpich-5.0.0 (parallel HDF5 with MPICH)
- **MPI**: /Users/guo.2154/local/mpich-5.0.0
- **Python**: /Users/guo.2154/local/Python-3.10.6-openssl-1.1.1q
- **CMake**: /Users/guo.2154/local/cmake-4.2.3
- **C++17 compiler**: Available via MPI wrappers

## Environment Setup

Set these paths in your shell (add to ~/.bashrc or ~/.zshrc):

```bash
# MPI
export PATH=/Users/guo.2154/local/mpich-5.0.0/bin:$PATH
export LD_LIBRARY_PATH=/Users/guo.2154/local/mpich-5.0.0/lib:$LD_LIBRARY_PATH

# HDF5
export HDF5_ROOT=$HOME/local/hdf5-1.14.6-mpich-5.0.0
export PATH=$HDF5_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH

# Python
export PATH=/Users/guo.2154/local/Python-3.10.6-openssl-1.1.1q/bin:$PATH
```

## Installation Steps

### 1. Install Henson

```bash
# Clone Henson
git clone https://github.com/henson-insitu/henson.git
cd henson
mkdir build && cd build

# Configure with your MPI
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/henson \
  -DCMAKE_CXX_COMPILER=/Users/guo.2154/local/mpich-5.0.0/bin/mpicxx \
  -DCMAKE_C_COMPILER=/Users/guo.2154/local/mpich-5.0.0/bin/mpicc

# Build and install
make -j8
make install

# Set environment variable (add to ~/.bashrc or ~/.zshrc)
export HENSON=$HOME/local/henson
```

### 2. Install LowFive

```bash
# Clone LowFive
git clone https://github.com/diatomic/LowFive.git
cd LowFive
mkdir build && cd build

# Configure with your parallel HDF5
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/lowfive \
  -DCMAKE_CXX_COMPILER=/Users/guo.2154/local/mpich-5.0.0/bin/mpicxx \
  -DCMAKE_C_COMPILER=/Users/guo.2154/local/mpich-5.0.0/bin/mpicc \
  -DHDF5_ROOT=$HOME/local/hdf5-1.14.6-mpich-5.0.0

# Build and install
make -j8
make install
```

### 3. Install Wilkins

```bash
# Clone Wilkins
git clone https://github.com/orcunyildiz/wilkins.git
cd wilkins
mkdir build && cd build

# Configure with your installations
cmake .. \
  -DCMAKE_CXX_COMPILER=/Users/guo.2154/local/mpich-5.0.0/bin/mpicxx \
  -DCMAKE_C_COMPILER=/Users/guo.2154/local/mpich-5.0.0/bin/mpicc \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/wilkins \
  -DHENSON_INCLUDE_DIR=$HOME/local/henson/include \
  -DHENSON_LIBRARY=$HOME/local/henson/lib/libhenson.a \
  -DHENSON_PMPI_LIBRARY=$HOME/local/henson/lib/libhenson-pmpi-static.a \
  -DPYTHON_EXECUTABLE=/Users/guo.2154/local/Python-3.10.6-openssl-1.1.1q/bin/python3 \
  -DLOWFIVE_INCLUDE_DIR=$HOME/local/lowfive/include \
  -DLOWFIVE_LIBRARY=$HOME/local/lowfive/lib/liblowfive.dylib \
  -DLOWFIVE_DIST_LIBRARY=$HOME/local/lowfive/lib/liblowfive-dist.a \
  -DHDF5_ROOT=$HOME/local/hdf5-1.14.6-mpich-5.0.0

# Build and install
make -j8
make install

# Add wilkins-master.py to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=$HOME/local/wilkins/bin:$PATH
```

### 4. Configure ndarray with Henson Support

Once Henson is installed, rebuild ndarray with Henson enabled:

```bash
cd /Users/guo.2154/workspace/projects/ndarray
mkdir build_wilkins && cd build_wilkins

cmake .. \
  -DCMAKE_CXX_COMPILER=/Users/guo.2154/local/mpich-5.0.0/bin/mpicxx \
  -DCMAKE_C_COMPILER=/Users/guo.2154/local/mpich-5.0.0/bin/mpicc \
  -DNDARRAY_USE_HDF5=TRUE \
  -DNDARRAY_USE_HENSON=TRUE \
  -DNDARRAY_USE_MPI=TRUE \
  -DNDARRAY_BUILD_TESTS=ON \
  -DNDARRAY_BUILD_EXAMPLES=ON \
  -DHDF5_ROOT=$HOME/local/hdf5-1.14.6-mpich-5.0.0 \
  -DHENSON=$HOME/local/henson

make -j8
```

## Runtime Environment for Wilkins

Before running Wilkins workflows, set these environment variables:

```bash
export HDF5_VOL_CONNECTOR="lowfive under_vol=0;under_info={};"
export HDF5_PLUGIN_PATH=$HOME/local/lowfive/lib
```

## Testing Wilkins

Run Wilkins example workflows:

```bash
cd $HOME/local/wilkins/examples/lowfive/cycle
./run_cycle.sh
```

## Using ndarray with Wilkins Workflows

Your ndarray tasks need to be:

1. **Position-independent code** - Already configured in src/CMakeLists.txt:1-8
2. **Linked with Henson** - Already configured when NDARRAY_USE_HENSON=TRUE
3. **Built as shared objects** - The ndarray library is already built as SHARED

### Example Workflow Configuration

Create a `workflow.yaml` file for Wilkins:

```yaml
# Example Wilkins workflow configuration
tasks:
  - name: producer
    exec: /path/to/your/producer_executable
    ranks: 2

  - name: consumer
    exec: /path/to/your/consumer_executable
    ranks: 2

connections:
  - from: producer
    to: consumer
    dataset: /data/field
```

### Running Your Workflow

```bash
mpirun -n 4 python wilkins-master.py workflow.yaml
```

## Verification Steps

1. Check HDF5 version:
```bash
h5cc -showconfig | grep Version
```

2. Verify Henson installation:
```bash
ls $HENSON/lib/libhenson.a
```

3. Verify LowFive installation:
```bash
ls /path/to/lowfive/install/lib/liblowfive.dylib
```

4. Test ndarray with Henson:
```bash
cd build_wilkins
make test
```

## Troubleshooting

- **HDF5 version**: You have HDF5 1.14.6 built with MPICH 5.0.0 (parallel support included)
- **Linker flags**: The Henson linker flags are already configured in ndarray's src/CMakeLists.txt:1-8
- **Library not found errors**: Make sure all paths are in LD_LIBRARY_PATH as shown in Environment Setup

## References

- [Wilkins GitHub](https://github.com/orcunyildiz/wilkins)
- [LowFive GitHub](https://github.com/diatomic/LowFive)
- [Henson GitHub](https://github.com/henson-insitu/henson)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
