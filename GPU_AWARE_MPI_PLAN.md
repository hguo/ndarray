# GPU-Aware MPI Implementation Plan

## Goal

Enable `exchange_ghosts()` to work when ndarray data is on GPU, supporting direct device-to-device transfers when MPI implementation allows.

## Design

### 1. Detection and Configuration

**Runtime Detection:**
```cpp
bool has_gpu_aware_mpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  return true;
#elif defined(MPIX_Query_cuda_support)
  return MPIX_Query_cuda_support() == 1;
#else
  // Check via environment variable (OpenMPI, MVAPICH2)
  const char* env = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
  if (env && std::string(env) == "1") return true;

  env = std::getenv("OMPI_MCA_opal_cuda_support");
  if (env && std::string(env) == "true") return true;

  return false;  // Conservative default
#endif
}
```

**CMake Detection** (optional, for build-time optimization):
```cmake
option(NDARRAY_GPU_AWARE_MPI "Enable GPU-aware MPI support" ON)
```

### 2. Modified exchange_ghosts() Logic

```cpp
void exchange_ghosts() {
  if (!is_distributed()) return;

  if (is_on_device()) {
    // GPU path
    if (has_gpu_aware_mpi()) {
      exchange_ghosts_gpu_direct();  // Pass device pointers to MPI
    } else {
      exchange_ghosts_gpu_staged();  // Host staging
    }
  } else {
    // CPU path (current implementation)
    exchange_ghosts_cpu();
  }
}
```

### 3. Three Implementation Paths

#### Path A: CPU (Current - Already Works)
```cpp
void exchange_ghosts_cpu() {
  // Current implementation with host buffers
  std::vector<T> send_buffers, recv_buffers;
  // pack_boundary_data() reads from host memory
  // MPI_Send/Recv with host pointers
  // unpack_ghost_data() writes to host memory
}
```

#### Path B: GPU Direct (New - Best Performance)
```cpp
void exchange_ghosts_gpu_direct() {
  // Allocate device buffers
  T* d_send_buffers[num_neighbors];
  T* d_recv_buffers[num_neighbors];

  for (each neighbor) {
    cudaMalloc(&d_send_buffers[i], send_count * sizeof(T));
    cudaMalloc(&d_recv_buffers[i], recv_count * sizeof(T));
  }

  // Pack on device (CUDA kernel)
  for (each neighbor) {
    pack_boundary_data_kernel<<<...>>>(
      d_send_buffers[i], device_ptr, ...);
  }
  cudaDeviceSynchronize();

  // MPI with device pointers (GPU-aware MPI)
  for (each neighbor) {
    MPI_Irecv(d_recv_buffers[i], ...);  // Device pointer!
  }
  for (each neighbor) {
    MPI_Send(d_send_buffers[i], ...);   // Device pointer!
  }
  MPI_Waitall(...);

  // Unpack on device (CUDA kernel)
  for (each neighbor) {
    unpack_ghost_data_kernel<<<...>>>(
      device_ptr, d_recv_buffers[i], ...);
  }
  cudaDeviceSynchronize();

  // Cleanup
  for (each neighbor) {
    cudaFree(d_send_buffers[i]);
    cudaFree(d_recv_buffers[i]);
  }
}
```

#### Path C: GPU Staged (New - Fallback)
```cpp
void exchange_ghosts_gpu_staged() {
  // Strategy: GPU → host → MPI → host → GPU

  // 1. Copy device data to host
  copy_from_device();  // Full array copy

  // 2. Exchange on host
  exchange_ghosts_cpu();

  // 3. Copy back to device
  copy_to_device();  // Full array copy
}
```

**Optimization for Path C:**
Only copy ghost+boundary regions instead of full array:
```cpp
void exchange_ghosts_gpu_staged_optimized() {
  // Allocate host buffers for boundary+ghost regions only
  std::vector<T> h_boundary, h_ghost;

  // Copy boundary from device to host
  for (each boundary region) {
    cudaMemcpy2D(...);  // Copy boundary slice
  }

  // MPI exchange
  exchange_ghosts_cpu();

  // Copy ghosts from host back to device
  for (each ghost region) {
    cudaMemcpy2D(...);  // Copy ghost slice
  }
}
```

### 4. CUDA Kernels for Packing/Unpacking

```cuda
// pack_boundary_kernel.cu
template <typename T>
__global__ void pack_boundary_2d_kernel(
    T* buffer,           // Output: packed buffer
    const T* array,      // Input: full array on device
    int dim0, int dim1,  // Array dimensions
    int boundary_dim,    // 0 or 1
    bool is_high,        // left/right or up/down
    int ghost_width,
    int core_size0, int core_size1)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (boundary_dim == 0) ? (ghost_width * core_size1)
                                   : (core_size0 * ghost_width);

  if (idx >= total) return;

  if (boundary_dim == 0) {
    // Boundary in dimension 0
    int i = idx / core_size1;
    int j = idx % core_size1;

    int src_i = is_high ? (core_size0 - ghost_width + i) : i;
    buffer[idx] = array[src_i * dim1 + j];
  } else {
    // Boundary in dimension 1
    int i = idx / ghost_width;
    int j = idx % ghost_width;

    int src_j = is_high ? (core_size1 - ghost_width + j) : j;
    buffer[idx] = array[i * dim1 + src_j];
  }
}

template <typename T>
__global__ void unpack_ghost_2d_kernel(
    T* array,            // Output: full array on device
    const T* buffer,     // Input: received ghost data
    int dim0, int dim1,
    int ghost_dim,
    bool is_high,
    int ghost_width,
    int core_size0, int core_size1)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (ghost_dim == 0) ? (ghost_width * core_size1)
                                : (core_size0 * ghost_width);

  if (idx >= total) return;

  if (ghost_dim == 0) {
    int i = idx / core_size1;
    int j = idx % core_size1;

    int dst_i = is_high ? (core_size0 + i) : i;  // Ghost region
    array[dst_i * dim1 + j] = buffer[idx];
  } else {
    int i = idx / ghost_width;
    int j = idx % ghost_width;

    int dst_j = is_high ? (core_size1 + j) : j;
    array[i * dim1 + dst_j] = buffer[idx];
  }
}
```

### 5. File Organization

**New files:**
- `include/ndarray/ndarray_mpi_gpu.hh` - GPU-aware MPI helpers
- `src/ndarray_mpi_gpu.cu` - CUDA kernels for pack/unpack (if separate compilation)

**Modified files:**
- `include/ndarray/ndarray.hh` - Add exchange_ghosts_gpu_* methods
- `CMakeLists.txt` - Add CUDA kernel compilation if needed

### 6. API Additions

No user-facing API changes! Everything automatic:
```cpp
// User code remains unchanged
temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});
temp.read_netcdf_auto("input.nc", "temp");
temp.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

// Now works with GPU data!
temp.exchange_ghosts();  // Automatically uses GPU path

// Run kernels
my_stencil_kernel<<<...>>>(temp.get_devptr(), ...);
```

### 7. Configuration Options

Users can control behavior via environment variables:
```bash
# Force host staging even if GPU-aware MPI available
export NDARRAY_FORCE_HOST_STAGING=1

# Disable GPU-aware MPI detection
export NDARRAY_DISABLE_GPU_AWARE_MPI=1

# Run normally (auto-detect)
mpirun -np 4 ./my_program
```

### 8. Testing Strategy

**Unit tests:**
1. Test GPU-aware MPI detection
2. Test pack_boundary_kernel correctness
3. Test unpack_ghost_kernel correctness
4. Test exchange_ghosts() with device data (2-4 ranks)
5. Test fallback to host staging

**Integration tests:**
- Distributed stencil on GPU
- Multi-GPU heat diffusion
- Compare CPU vs GPU ghost exchange results

### 9. Performance Considerations

**GPU Direct (Best):**
- No host staging overhead
- ~16 GB/s PCIe bandwidth saved (x2 for round-trip)
- Kernel launch overhead (~5 μs) negligible vs MPI latency

**GPU Staged (Fallback):**
- 2x full array copies (device ↔ host)
- For 1000×800 float array: ~3 MB × 2 = 6 MB transfer
- At 12 GB/s PCIe: ~0.5 ms overhead

**Optimization:**
- Staged mode should only copy boundary+ghost regions
- For 1000×800 with 1-layer ghosts: ~7 KB instead of 3 MB
- 400x less data movement!

### 10. Documentation

**Add to docs/GPU_SUPPORT.md:**
- Section on distributed GPU arrays
- GPU-aware MPI requirements
- Performance comparison table
- Troubleshooting GPU + MPI

**Update docs/DISTRIBUTED_NDARRAY.md:**
- Mark GPU-aware MPI as ✅ implemented
- Add usage examples

### 11. Implementation Phases

**Phase 1: Detection and Infrastructure**
- Add has_gpu_aware_mpi() detection
- Add exchange_ghosts_gpu_staged() with full-array staging
- Test with GPU data

**Phase 2: GPU Direct Path**
- Implement CUDA kernels for pack/unpack
- Implement exchange_ghosts_gpu_direct()
- Test with GPU-aware MPI

**Phase 3: Optimization**
- Optimize staged mode (partial copy)
- Benchmark and tune kernel parameters
- Add performance tests

**Phase 4: Documentation and Examples**
- Update documentation
- Add distributed GPU example
- Add performance guide

## Timeline

- Phase 1: 2-3 hours (detection + staged fallback)
- Phase 2: 3-4 hours (GPU direct with kernels)
- Phase 3: 2-3 hours (optimization)
- Phase 4: 1-2 hours (docs)

**Total: ~8-12 hours for complete implementation**

## Success Criteria

✅ exchange_ghosts() works when data is on GPU
✅ Auto-detects GPU-aware MPI at runtime
✅ Falls back to host staging gracefully
✅ No API changes (transparent to users)
✅ Tests pass with 2-4 ranks on GPU
✅ Performance comparable to CPU-only case
✅ Documentation complete

## Questions for User

1. Which MPI implementation do you use? (OpenMPI, MPICH, Cray MPI, etc.)
2. Do you have GPU-aware MPI available? (Can check with: `ompi_info --parsable --all | grep mpi_built_with_cuda_support`)
3. Priority: Full implementation or just staged fallback first?
4. Target: CUDA only, or also HIP/ROCm for AMD GPUs?
