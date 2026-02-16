# GPU Support Guide

## Overview

ndarray provides explicit GPU memory management for CUDA and SYCL devices. The design follows HPC best practices with explicit data transfers, allowing users to control when expensive host-device transfers occur.

## Supported Devices

- **CUDA** - NVIDIA GPUs (requires CUDA Toolkit)
- **SYCL** - Cross-platform accelerators (Intel, AMD, NVIDIA)

## Building with GPU Support

### CUDA

```bash
cmake .. \
  -DNDARRAY_USE_CUDA=ON \
  -DCUDAToolkit_ROOT=/path/to/cuda
make
```

### SYCL

```bash
cmake .. \
  -DNDARRAY_USE_SYCL=ON \
  -DSYCL_IMPLEMENTATION=dpcpp  # or hipsycl, computecpp
make
```

## API Reference

### Device Management

```cpp
// Move data to device (clears host memory)
arr.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);  // device_type, device_id

// Move data back to host (frees device memory)
arr.to_host();

// Copy to device (keeps host memory)
arr.copy_to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

// Copy from device (keeps device memory)
arr.copy_from_device();

// Query device status
bool on_host = arr.is_on_host();
bool on_device = arr.is_on_device();
int dev_type = arr.get_device_type();  // NDARRAY_DEVICE_HOST/CUDA/SYCL
int dev_id = arr.get_device_id();
void* dev_ptr = arr.get_devptr();  // Raw device pointer for kernels
```

### Device Types

```cpp
ftk::NDARRAY_DEVICE_HOST   // 0 - CPU memory
ftk::NDARRAY_DEVICE_CUDA   // 1 - NVIDIA GPU
ftk::NDARRAY_DEVICE_HIP    // 2 - AMD GPU (reserved)
ftk::NDARRAY_DEVICE_SYCL   // 3 - SYCL device
```

## Usage Patterns

### Pattern 1: HPC I/O Workflow

Typical workflow for HPC scientific applications:

```cpp
#include <ndarray/ndarray.hh>

int main() {
  // 1. Read data on host (I/O is CPU-bound)
  ftk::ndarray<float> arr;
  arr.read_netcdf("input.nc", "temperature");

  // 2. Move to GPU (explicit transfer)
  arr.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

  // 3. Get device pointer for your kernels
  float* d_ptr = static_cast<float*>(arr.get_devptr());

  // 4. Run your CUDA kernel
  my_kernel<<<blocks, threads>>>(d_ptr, arr.size());
  cudaDeviceSynchronize();

  // 5. Move back to host (explicit transfer)
  arr.to_host();

  // 6. Write results (I/O is CPU-bound)
  arr.write_netcdf("output.nc", "result");

  return 0;
}
```

**Key points:**
- All I/O (NetCDF, HDF5, ADIOS2) happens on host
- Transfers are explicit and controlled
- No hidden performance costs
- Compatible with existing HPC codes

### Pattern 2: Copy Semantics (Debug/Visualization)

Keep data on both host and device:

```cpp
// Copy to device (host data preserved)
arr.copy_to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

// Now data is on both host and device
assert(arr.size() > 0);           // Host data accessible
assert(arr.get_devptr() != nullptr);  // Device data available

// Run kernel on device
my_kernel<<<blocks, threads>>>(
    static_cast<float*>(arr.get_devptr()),
    arr.size());

// Peek at results without stopping device work
arr.copy_from_device();

// Host data now has device results, device memory still allocated
std::cout << "First result: " << arr[0] << std::endl;

// Continue GPU work...

// Final cleanup
arr.to_host();  // Frees device memory
```

**Use cases:**
- Debugging (inspect intermediate results)
- Visualization (stream data to CPU while GPU computes)
- Overlapping compute and I/O

### Pattern 3: Multi-GPU

Distribute work across multiple GPUs:

```cpp
#include <ndarray/ndarray_cuda.hh>

int main() {
  int num_gpus = ftk::nd::get_cuda_device_count();
  std::vector<ftk::ndarray<float>> arrays(num_gpus);

  // Load and distribute data
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    arrays[gpu].read_netcdf("input.nc", "temperature",
                            /*offset=*/gpu * chunk_size,
                            /*count=*/chunk_size);

    arrays[gpu].to_device(ftk::NDARRAY_DEVICE_CUDA, gpu);
  }

  // Process on each GPU (could use OpenMP here)
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu);
    my_kernel<<<blocks, threads>>>(
        static_cast<float*>(arrays[gpu].get_devptr()),
        arrays[gpu].size());
  }

  // Gather results
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    arrays[gpu].to_host();
  }

  return 0;
}
```

## CUDA-Specific Features

### Device Queries

```cpp
#include <ndarray/ndarray_cuda.hh>

// Check device availability
bool has_cuda = ftk::nd::cuda_device_available(0);

// Get device count
int num_devices = ftk::nd::get_cuda_device_count();

// Get device properties
cudaDeviceProp prop = ftk::nd::get_cuda_device_properties(0);
std::cout << "Device: " << prop.name << std::endl;
std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
std::cout << "Total memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;

// Print device info
ftk::nd::print_cuda_device_info(0);
```

### Error Handling

CUDA API calls are wrapped with `CUDA_CHECK` macro:

```cpp
// Automatic error checking
arr.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);  // Throws ftk::nd::device_error on failure

// Manual error checking
try {
  arr.to_device(ftk::NDARRAY_DEVICE_CUDA, 99);  // Invalid device
} catch (const ftk::nd::device_error& e) {
  std::cerr << "Device error: " << e.what() << std::endl;
  // Handle error gracefully
}
```

## SYCL-Specific Features

### Custom Queue

Provide your own SYCL queue for better control:

```cpp
#include <CL/sycl.hpp>

sycl::queue my_queue(sycl::gpu_selector{});

ftk::ndarray<float> arr;
arr.set_sycl_queue(&my_queue);

arr.to_device(ftk::NDARRAY_DEVICE_SYCL, 0);
// Uses your queue for memory operations
```

## Performance Considerations

### Transfer Bandwidth

Typical PCIe 3.0 x16 bandwidth: ~12 GB/s

```cpp
// Measure transfer performance
auto start = std::chrono::high_resolution_clock::now();
arr.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

std::cout << "Transfer time: " << duration.count() << " ms" << std::endl;
std::cout << "Bandwidth: " << (arr.size() * sizeof(float) / duration.count()) / 1e6 << " GB/s" << std::endl;
```

### Minimize Transfers

```cpp
// BAD: Transfer in loop
for (int iter = 0; iter < 100; iter++) {
  arr.to_device(...);
  // kernel
  arr.to_host();
}

// GOOD: Transfer once
arr.to_device(...);
for (int iter = 0; iter < 100; iter++) {
  // kernel
}
arr.to_host();
```

### Use Pinned Memory (Advanced)

For maximum transfer performance, consider using CUDA pinned memory for the host array (requires custom allocator - not currently supported but could be added).

## Integration with External Libraries

### PyTorch Tensors

```cpp
// ndarray to PyTorch
torch::Tensor to_torch(const ftk::ndarray<float>& arr) {
  std::vector<int64_t> shape(arr.nd());
  for (size_t i = 0; i < arr.nd(); i++) {
    shape[i] = arr.dimf(i);
  }

  return torch::from_blob(
      const_cast<float*>(arr.data()),
      shape,
      torch::kFloat32).clone();  // Clone to own the data
}

// PyTorch to ndarray
ftk::ndarray<float> from_torch(torch::Tensor t) {
  ftk::ndarray<float> arr;
  std::vector<size_t> dims;
  for (int64_t d : t.sizes()) {
    dims.push_back(d);
  }
  arr.reshapef(dims);

  std::memcpy(arr.data(), t.data_ptr<float>(), arr.size() * sizeof(float));
  return arr;
}
```

### CuPy Arrays

```python
import cupy as cp
import ctypes

# ndarray device pointer to CuPy
def ndarray_to_cupy(devptr, shape, dtype=cp.float32):
    size = np.prod(shape)
    mem = cp.cuda.UnownedMemory(devptr, size * dtype().itemsize, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(shape, dtype=dtype, memptr=memptr)
```

## Best Practices

### ✅ DO

1. **Profile before optimizing** - Measure to find actual bottlenecks
2. **Minimize transfers** - Keep data on device as long as possible
3. **Use explicit transfers** - Clear intent, predictable performance
4. **Check device availability** - Not all systems have GPUs
5. **Handle errors** - Use try-catch for robust code

```cpp
// Check availability first
if (ftk::nd::cuda_device_available(0)) {
  arr.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
  // GPU code
  arr.to_host();
} else {
  // CPU fallback
}
```

### ❌ DON'T

1. **Don't transfer in loops** - Very slow
2. **Don't assume GPU availability** - Check first
3. **Don't mix host and device access** - Undefined behavior
4. **Don't forget to synchronize** - Kernels are asynchronous

```cpp
// BAD
arr.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
float val = arr[0];  // Host data was cleared!

// GOOD
arr.copy_to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
float val = arr[0];  // Host data still valid
```

## Troubleshooting

### Common Issues

**Problem**: `CUDA_ERROR_INVALID_DEVICE`
- **Solution**: Check device ID is valid (< device count)

**Problem**: Segmentation fault when accessing array
- **Solution**: Make sure data is on host before accessing with `arr[i]`

**Problem**: Out of memory
- **Solution**: Use `to_device()` (move semantics) instead of `copy_to_device()`

## Testing

Run GPU tests:

```bash
cd build
make test_gpu
./bin/test_gpu
```

Expected output:
```
=== Running GPU Tests ===

CUDA support: ENABLED
CUDA devices found: 1
CUDA Device 0: NVIDIA GeForce RTX 3080
  Compute capability: 8.6
  Total global memory: 10240 MB
  ...

  Testing: to_device() and to_host() - move semantics
    PASSED
  Testing: copy_to_device() and copy_from_device() - copy semantics
    PASSED
  ...

=== All GPU tests passed ===
```

## Distributed GPU Arrays (MPI + GPU)

**NEW**: exchange_ghosts() now works with GPU data!

### GPU-Aware MPI Support

The library automatically detects GPU-aware MPI and uses direct device-to-device transfers:

```cpp
#include <ndarray/ndarray.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  ftk::ndarray<float> temp;

  // Domain decomposition
  temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});

  // Read data (on host)
  temp.read_netcdf_auto("input.nc", "temperature");

  // Move to GPU
  temp.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

  // Ghost exchange now works on GPU!
  temp.exchange_ghosts();

  // Run your CUDA kernel
  my_stencil_kernel<<<blocks, threads>>>(
      static_cast<float*>(temp.get_devptr()),
      temp.size());

  MPI_Finalize();
  return 0;
}
```

### Three Automatic Paths

The library chooses the appropriate path automatically based on runtime detection:

1. **GPU Direct**:
   - Uses GPU-aware MPI for device-to-device transfers
   - CUDA kernels pack/unpack data directly on device
   - No host staging required
   - Requires: GPU-aware MPI implementation

2. **GPU Staged** (Fallback):
   - Stages through host memory (GPU → host → MPI → host → GPU)
   - Works with any MPI implementation
   - Automatic when GPU-aware MPI not available

3. **CPU Path**:
   - Traditional host-based exchange
   - Used when data is on host

### GPU-Aware MPI Detection

Automatic runtime detection checks:
- Compile-time macros (`MPIX_CUDA_AWARE_SUPPORT`)
- Runtime queries (`MPIX_Query_cuda_support()`)
- Environment variables (MPICH, OpenMPI, Cray MPI)

### Checking GPU-Aware MPI

**OpenMPI:**
```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support
```

**MPICH/MVAPICH2:**
```bash
echo $MPICH_GPU_SUPPORT_ENABLED
```

**At runtime:**
```bash
# Verify it's being used (library will auto-detect)
mpirun -np 4 ./my_program
```

### Environment Variables

Control behavior with environment variables:

```bash
# Force host staging even if GPU-aware MPI available
export NDARRAY_FORCE_HOST_STAGING=1

# Disable GPU-aware MPI detection entirely
export NDARRAY_DISABLE_GPU_AWARE_MPI=1

# Normal operation (auto-detect)
mpirun -np 4 ./my_program
```

### Best Practices

✅ **DO:**
- Let the library auto-detect (no code changes needed)
- Test with GPU-aware MPI if available
- Verify ghost exchange works correctly for your workflow

❌ **DON'T:**
- Manually copy to host before `exchange_ghosts()` (automatic now!)
- Assume GPU-aware MPI is available (library handles fallback)

### Example: Distributed Heat Diffusion on GPU

```cpp
#include <ndarray/ndarray.hh>
#include <mpi.h>

__global__ void heat_diffusion_kernel(float* T, const float* T_old,
                                       int nx, int ny, float alpha, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // +1 to skip ghost
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i < nx-1 && j < ny-1) {
    float laplacian = (T_old[(i-1)*ny + j] + T_old[(i+1)*ny + j] +
                       T_old[i*ny + (j-1)] + T_old[i*ny + (j+1)] -
                       4.0f * T_old[i*ny + j]);
    T[i*ny + j] = T_old[i*ny + j] + alpha * dt * laplacian;
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  ftk::ndarray<float> T, T_old;

  // Decompose with ghosts
  T.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});
  T_old.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});

  // Initialize on host, move to GPU
  // ... initialization ...
  T.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
  T_old.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

  // Time-stepping
  for (int step = 0; step < 1000; step++) {
    // Exchange ghosts (automatic GPU path!)
    T_old.exchange_ghosts();

    // Compute on GPU
    dim3 threads(16, 16);
    dim3 blocks((T.dim(0) + 15) / 16, (T.dim(1) + 15) / 16);

    heat_diffusion_kernel<<<blocks, threads>>>(
        static_cast<float*>(T.get_devptr()),
        static_cast<float*>(T_old.get_devptr()),
        T.dim(0), T.dim(1), 0.1f, 0.01f);

    std::swap(T, T_old);
  }

  // Move back to host for I/O
  T.to_host();
  T.write_netcdf_auto("output.nc", "temperature");

  MPI_Finalize();
  return 0;
}
```

### Troubleshooting

**Problem**: "GPU-aware MPI not detected but I have it"
- **Solution**: Check with `ompi_info` or similar, set environment variables if needed

**Problem**: Segmentation fault in exchange_ghosts()
- **Solution**: Ensure data is on device before calling, check GPU-aware MPI is working correctly

## See Also

- [EXCEPTION_HANDLING.md](EXCEPTION_HANDLING.md) - Error handling with exceptions
- [ZERO_COPY_OPTIMIZATION.md](ZERO_COPY_OPTIMIZATION.md) - Memory optimization
- [examples/device_memory.cpp](../examples/device_memory.cpp) - Complete example
