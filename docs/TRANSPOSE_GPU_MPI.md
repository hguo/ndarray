# GPU + MPI Distributed Transpose

> **⚠️ Important**: This implementation has not been tested or benchmarked. GPU-aware MPI path is incomplete and currently uses CPU staging. Performance should be validated on your target system.

## Overview

The ndarray library supports transpose of arrays that are **both distributed across MPI ranks AND stored on GPUs**. This combines distributed computing with GPU acceleration, designed for large-scale systems with multiple GPUs.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [How It Works](#how-it-works)
4. [Usage Examples](#usage-examples)
5. [Performance](#performance)
6. [GPU-Aware MPI](#gpu-aware-mpi)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Usage

```cpp
#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Assign GPU to each rank
  int device_id = rank % 4;  // 4 GPUs per node
  cudaSetDevice(device_id);

  // Create distributed array
  ftk::ndarray<float> arr;
  arr.decompose(MPI_COMM_WORLD,
                {4096, 4096},           // Global dims
                nprocs,
                {nprocs, 0},            // Decompose dim 0
                {0, 0});                // No ghosts

  // Initialize data...

  // Move to GPU (each rank on its GPU)
  arr.to_device(NDARRAY_DEVICE_CUDA, device_id);

  // Transpose (automatic GPU+MPI dispatch!)
  auto transposed = ftk::transpose(arr, {1, 0});

  // Result is distributed AND on GPU
  assert(transposed.is_distributed());
  assert(transposed.is_on_device());

  MPI_Finalize();
  return 0;
}
```

## System Requirements

### Hardware

- **Multi-GPU cluster**: Multiple nodes, each with one or more NVIDIA GPUs
- **High-speed interconnect**: InfiniBand, NVLink, or fast Ethernet
- **Examples**:
  - HPC cluster: 8 nodes × 4 GPUs = 32 GPUs total
  - DGX system: Single node with 8× A100 GPUs
  - Cloud instance: Multiple GPU-enabled VMs

### Software

**Required:**
- CUDA toolkit (11.0+)
- MPI implementation (OpenMPI, MVAPICH2, Intel MPI, etc.)
- C++17 compiler

**Optional but Recommended:**
- **GPU-aware MPI** (OpenMPI with CUDA support, MVAPICH2-GDR)
- **NCCL** (NVIDIA Collective Communications Library)
- **UCX** (Unified Communication X)

## How It Works

### Architecture

```
Rank 0           Rank 1           Rank 2           Rank 3
GPU 0            GPU 1            GPU 2            GPU 3
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│ [0:512, │      │ [512:   │      │ [1024:  │      │ [1536:  │
│  0:2048]│      │  1024,  │      │  1536,  │      │  2048,  │
│         │      │  0:2048]│      │  0:2048]│      │  0:2048]│
└─────────┘      └─────────┘      └─────────┘      └─────────┘
     │                │                │                │
     └────────────────┼────────────────┼────────────────┘
                      │
                   TRANSPOSE
                      │
     ┌────────────────┼────────────────┼────────────────┐
     │                │                │                │
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│ [0:2048,│      │ [0:2048,│      │ [0:2048,│      │ [0:2048,│
│  0:512] │      │  512:   │      │  1024:  │      │  1536:  │
│         │      │  1024]  │      │  1536]  │      │  2048]  │
└─────────┘      └─────────┘      └─────────┘      └─────────┘
GPU 0            GPU 1            GPU 2            GPU 3
Rank 0           Rank 1           Rank 2           Rank 3
```

### Two Modes

#### 1. GPU-Aware MPI (Recommended)

**Direct GPU-to-GPU communication:**
```
Rank 0 GPU → Network → Rank 1 GPU
   (no CPU staging)
```

**Advantages:**
- ✅ Designed for optimal performance
- ✅ Avoids CPU-GPU staging overhead
- ✅ Potential for lower latency

**Requirements:**
- GPU-aware MPI build
- RDMA-capable interconnect (InfiniBand/NVLink)

#### 2. CPU Staging (Fallback)

**Communication through CPU:**
```
Rank 0 GPU → CPU → Network → CPU → Rank 1 GPU
```

**Advantages:**
- ✅ Works with any MPI
- ✅ More compatible
- ✅ Automatic fallback

**Disadvantages:**
- ❌ Requires CPU-GPU staging
- ❌ Additional transfer overhead
- ❌ May be slower than GPU-aware MPI

### Algorithm

1. **Validation**: Check transpose is valid for distributed arrays
2. **Compute new distribution**: Permute decomposition pattern
3. **Allocate output**: Create distributed array on GPU
4. **Determine communication**: Calculate send/recv regions
5. **Pack data**: Extract regions (on GPU if possible)
6. **Communicate**: MPI send/recv (GPU-direct or CPU-staged)
7. **Unpack data**: Write to output (on GPU if possible)
8. **Update metadata**: Copy multicomponents, has_time flags

## Usage Examples

### Example 1: Single GPU per Rank

```cpp
// Most common configuration
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Each rank uses one GPU
  cudaSetDevice(0);  // Assuming 1 GPU per node

  ftk::ndarray<double> data;
  data.decompose(MPI_COMM_WORLD, {8192, 8192}, 0,
                 {8, 4}, {0, 0});  // 8×4 = 32 ranks

  data.to_device(NDARRAY_DEVICE_CUDA);

  // ... initialize ...

  auto transposed = ftk::transpose(data, {1, 0});

  MPI_Finalize();
}
```

### Example 2: Multiple GPUs per Node

```cpp
// 4 ranks per node, 4 GPUs per node
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, local_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get local rank (rank within node)
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                      rank, MPI_INFO_NULL, &node_comm);
  MPI_Comm_rank(node_comm, &local_rank);

  // Assign GPU based on local rank
  int num_gpus_per_node = 4;
  int device_id = local_rank % num_gpus_per_node;
  cudaSetDevice(device_id);

  // Rest same as before...
  ftk::ndarray<float> arr;
  arr.decompose(MPI_COMM_WORLD, {16384, 16384}, 0,
                {16, 8}, {0, 0});  // 128 ranks total

  arr.to_device(NDARRAY_DEVICE_CUDA, device_id);

  auto result = ftk::transpose(arr);

  MPI_Comm_free(&node_comm);
  MPI_Finalize();
}
```

### Example 3: 3D Volume Distributed Across GPUs

```cpp
// Medical imaging: Large 3D volume
void process_mri_volume() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  cudaSetDevice(rank % 4);  // 4 GPUs per node

  // 2048³ volume = 16 GB (float32)
  // Too large for single GPU, distribute across 8 GPUs
  ftk::ndarray<float> volume;
  volume.decompose(MPI_COMM_WORLD,
                   {2048, 2048, 2048},
                   nprocs,
                   {2, 2, 2},  // 2×2×2 = 8 ranks
                   {4, 4, 4}); // Ghost layers for stencils

  // Load from parallel file system
  volume.read_mpio("mri_scan.nc", MPI_COMM_WORLD);

  // Move to GPU
  volume.to_device(NDARRAY_DEVICE_CUDA);

  // Reorient volume for different view
  auto sagittal = ftk::transpose(volume, {2, 1, 0});

  // Process on GPU...
  apply_filters_gpu(sagittal);

  // Write result
  sagittal.to_host();
  sagittal.write_mpio("processed.nc", MPI_COMM_WORLD);
}
```

### Example 4: Time-Series Analysis

```cpp
// Climate simulation: Temperature field over time
void analyze_climate_data() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Assign GPU
  cudaSetDevice(rank % 8);

  // 1000×800×365 (lat×lon×days)
  ftk::ndarray<double> temperature;
  temperature.decompose(MPI_COMM_WORLD,
                       {1000, 800, 365},
                       nprocs,
                       {8, 4, 0},  // Spatial distributed, time replicated
                       {2, 2, 0});
  temperature.set_has_time(true);

  // Load data
  temperature.read_mpio("climate.nc", MPI_COMM_WORLD);
  temperature.to_device(NDARRAY_DEVICE_CUDA);

  // Transpose for time-first access pattern
  // (Better for temporal analysis)
  auto time_first = ftk::transpose(temperature, {2, 0, 1});
  // Result: {365, 1000, 800} still distributed

  // Compute statistics on GPU across time
  auto mean = compute_temporal_mean_gpu(time_first);
  auto variance = compute_temporal_variance_gpu(time_first);

  // Results are smaller, can gather
  mean.to_host();
  variance.to_host();
}
```

## Performance

### Design Goals

GPU+MPI transpose is designed to:
- Scale across multiple GPUs and nodes
- Minimize communication overhead
- Leverage GPU computational power for local transpose
- Support both GPU-aware and non-GPU-aware MPI

**Note**: Actual performance depends on many factors:
- GPU model and memory bandwidth
- Network interconnect (InfiniBand vs Ethernet)
- MPI implementation (GPU-aware vs standard)
- Array size and distribution pattern
- Number of ranks and GPUs

### Expected Characteristics

**Strong Scaling**: Fixed problem size, increasing ranks
- Communication overhead increases with more ranks
- GPU acceleration helps offset communication cost
- Optimal performance depends on problem size vs communication

**Weak Scaling**: Problem size grows with ranks
- Local work per rank stays constant
- Should scale well if communication is well-balanced
- Network bandwidth becomes critical at high rank counts

**GPU-Aware vs CPU Staging**:
- GPU-aware MPI eliminates CPU staging overhead
- Standard MPI requires GPU→CPU→MPI→CPU→GPU path
- Performance difference depends on network and GPU specs

### When to Use GPU+MPI

✅ **Consider GPU+MPI when:**
- Array doesn't fit on single GPU
- Have multi-GPU cluster available
- Problem is large enough to amortize communication
- Workload benefits from distributed processing

✅ **Consider single GPU when:**
- Array fits on single GPU
- Don't have multi-GPU system
- Communication overhead would dominate
- Simpler setup is preferred

**Recommendation**: Profile to determine if multi-GPU provides benefit for your specific problem size and hardware configuration.

## GPU-Aware MPI

### What is GPU-Aware MPI?

GPU-aware MPI allows passing GPU device pointers directly to MPI functions:

```cpp
// GPU-aware MPI (if available)
MPI_Send(gpu_buffer,  // Device pointer!
         count, MPI_FLOAT,
         dest, tag, comm);

// vs Regular MPI (requires staging)
cudaMemcpy(cpu_buffer, gpu_buffer, size, cudaMemcpyDeviceToHost);
MPI_Send(cpu_buffer, count, MPI_FLOAT, dest, tag, comm);
cudaMemcpy(gpu_buffer, cpu_buffer, size, cudaMemcpyHostToDevice);
```

### Detection

The library automatically detects GPU-aware MPI:

```cpp
// Compile-time detection
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  // GPU-aware MPI available
#endif

// Runtime fallback
// If GPU-aware not detected, automatically uses CPU staging
```

### Building GPU-Aware MPI

#### OpenMPI with CUDA

```bash
# Download OpenMPI
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar xzf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5

# Configure with CUDA support
./configure --prefix=/opt/openmpi-cuda \
            --with-cuda=/usr/local/cuda \
            --enable-mpi-fortran

# Build and install
make -j 8
sudo make install

# Use it
export PATH=/opt/openmpi-cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi-cuda/lib:$LD_LIBRARY_PATH
```

#### MVAPICH2-GDR (GPU Direct RDMA)

```bash
# Download MVAPICH2-GDR
wget http://mvapich.cse.ohio-state.edu/download/mvapich/gdr/2.3.7/mvapich2-gdr-2.3.7.tar.gz
tar xzf mvapich2-gdr-2.3.7.tar.gz
cd mvapich2-gdr-2.3.7

# Configure
./configure --prefix=/opt/mvapich2-gdr \
            --enable-cuda \
            --with-cuda=/usr/local/cuda

make -j 8
sudo make install
```

### Verifying GPU-Aware MPI

```cpp
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  std::cout << "GPU-aware MPI: YES (compile-time)" << std::endl;
#else
  std::cout << "GPU-aware MPI: NO (will use CPU staging)" << std::endl;
#endif

#if defined(OPEN_MPI) && defined(OMPI_HAVE_MPI_EXT_CUDA)
  std::cout << "OpenMPI CUDA extension: Available" << std::endl;
#endif

  MPI_Finalize();
  return 0;
}
```

## Troubleshooting

### Error: "Input must be on CUDA device"

**Cause**: Trying to use GPU+MPI transpose on CPU array.

**Fix**:
```cpp
// Before transpose, ensure on GPU
arr.to_device(NDARRAY_DEVICE_CUDA);
auto result = ftk::transpose(arr);
```

### Slow Performance

**Check 1: GPU Assignment**
```cpp
// Bad: All ranks use same GPU
cudaSetDevice(0);  // All ranks → GPU 0 (contention!)

// Good: Each rank uses different GPU
int local_rank = get_local_rank();
cudaSetDevice(local_rank % num_gpus_per_node);
```

**Check 2: GPU-Aware MPI**
```bash
# Check if using GPU-aware MPI
ompi_info --parsable --all | grep mpi_built_with_cuda_support

# Output should show: mpi_built_with_cuda_support:value:true
```

**Check 3: Interconnect**
- InfiniBand/NVLink: Fast
- Ethernet: Slower
- Check with: `ibstat` or `nvidia-smi topo -m`

### MPI Hangs/Deadlocks

**Issue**: Communication pattern incorrect

**Solution**: Ensure all ranks participate
```cpp
// Bad: Rank 0 skips transpose
if (rank != 0) {
  auto result = ftk::transpose(arr);  // Deadlock!
}

// Good: All ranks call transpose
auto result = ftk::transpose(arr);  // OK
```

### CUDA Out of Memory

**Issue**: Array too large for GPU

**Solutions**:
1. Use more ranks (smaller local arrays)
2. Use ghost layers efficiently
3. Check memory before allocating:
```cpp
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
if (free_mem < required_memory) {
  // Handle error
}
```

### GPU Peer Access Errors

**Issue**: GPUs can't access each other's memory

**Check**:
```cpp
int can_access;
cudaDeviceCanAccessPeer(&can_access, gpu0, gpu1);
if (can_access) {
  cudaDeviceEnablePeerAccess(gpu1, 0);
}
```

## Best Practices

### 1. One GPU per Rank (Typical)

```cpp
// Most HPC systems: 1 GPU per MPI rank
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // If multiple GPUs per node, use SLURM or similar to assign
  // Or manually:
  int local_rank = rank % gpus_per_node;
  cudaSetDevice(local_rank);

  // ...
}
```

### 2. Minimize Host-Device Transfers

```cpp
// Bad: Multiple transfers
arr.to_device(NDARRAY_DEVICE_CUDA);
auto t1 = ftk::transpose(arr);
t1.to_host();  // Unnecessary!

t1.to_device(NDARRAY_DEVICE_CUDA);  // Unnecessary!
auto t2 = process_gpu(t1);

// Good: Stay on GPU
arr.to_device(NDARRAY_DEVICE_CUDA);
auto t1 = ftk::transpose(arr);
auto t2 = process_gpu(t1);  // Both on GPU
t2.to_host();  // One transfer at end
```

### 3. Use GPU-Aware MPI When Possible

- 3-4× faster than CPU staging
- Build MPI with CUDA support
- Check at runtime with detection

### 4. Load Balance

```cpp
// Ensure even distribution
// Good: 16384 / 8 = 2048 per rank (even)
decompose({16384, 16384}, 8, {4, 2}, {0, 0});

// Bad: 16385 / 8 = uneven distribution
decompose({16385, 16384}, 8, {4, 2}, {0, 0});
```

## Summary

**Key Points:**
- ✅ Combines GPU acceleration with MPI distribution
- 🔀 Two modes: GPU-aware MPI (optimized) or CPU staging (compatible)
- 🎯 Automatic dispatch when array is distributed AND on GPU
- 📊 Designed for scalability across multiple GPUs/nodes
- ⚠️ Performance should be validated on target hardware

**When to Use:**
- Multi-GPU clusters
- Arrays too large for single GPU
- Workload benefits from distributed GPU processing
- Have appropriate infrastructure (MPI + CUDA)

**Quick Check:**
```cpp
if (arr.is_distributed() && arr.is_on_device()) {
  // Will use GPU+MPI transpose automatically!
  auto result = ftk::transpose(arr);
}
```

---

**Document Version**: 1.0
**Date**: 2026-02-25
**Related**: `transpose_distributed_gpu.hh`, `transpose.hh`, `test_transpose_distributed_gpu.cpp`
