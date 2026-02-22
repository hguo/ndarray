# GPU Support Documentation

**Status**: Production-ready for data management
**Last Updated**: 2026-02-20

## Overview

The ndarray library provides GPU support for **data management** across multiple backends:
- **CUDA** (NVIDIA GPUs)
- **HIP** (AMD GPUs) - experimental
- **SYCL** (Cross-platform: Intel, NVIDIA, AMD) - experimental

**Scope**: GPU support focuses on **memory management and data transfers**, not compute kernels. The library provides:
- Device memory allocation/deallocation (RAII-based)
- Host ↔ Device data transfers
- Basic operations: `fill()`, `scale()`, `add()` on device
- GPU-aware MPI for distributed computing

## Quick Start

### 1. Build with CUDA Support

```bash
cmake -B build \
  -DNDARRAY_USE_CUDA=TRUE \
  -DNDARRAY_BUILD_TESTS=ON

cmake --build build
```

### 2. Basic Usage

```cpp
#include <ndarray/ndarray.hh>

using namespace ftk;

// Create array on host
ndarray<float> arr({1024, 1024});
arr.fill(1.0f);

// Move to GPU (transfers data, clears host memory)
arr.to_device(NDARRAY_DEVICE_CUDA);

// Perform operations on device
arr.scale(2.0f);  // Uses GPU kernel

// Move back to host
arr.to_host();  // Transfers data back, frees device memory
```

## API Reference

### Memory Management

#### `to_device(int device_type, int device_id = 0)`
Moves array to device, **clears host memory** to save RAM.

```cpp
arr.to_device(NDARRAY_DEVICE_CUDA);      // NVIDIA GPU
arr.to_device(NDARRAY_DEVICE_CUDA, 1);   // GPU 1 (multi-GPU)
arr.to_device(NDARRAY_DEVICE_SYCL);      // SYCL device
```

#### `to_host()`
Moves array back to host, **frees device memory**.

```cpp
arr.to_host();
```

#### `copy_to_device(int device_type, int device_id = 0)`
Copies array to device, **keeps host data**.

```cpp
arr.copy_to_device(NDARRAY_DEVICE_CUDA);
// Host data still accessible
```

#### `copy_from_device()`
Copies device data back to host, **keeps device memory**.

```cpp
arr.copy_from_device();
// Device data still valid
```

### Query Methods

```cpp
bool is_on_device() const;         // True if on GPU
bool is_on_host() const;           // True if on CPU
int get_device_type() const;       // NDARRAY_DEVICE_CUDA, etc.
int get_device_id() const;         // Device ID (0, 1, ...)
void* get_devptr();                // Raw device pointer (advanced)
```

### Device Operations

These operations execute on GPU when array is on device:

#### `fill(T value)`
Fill array with constant value.

```cpp
arr.to_device(NDARRAY_DEVICE_CUDA);
arr.fill(3.14f);  // GPU kernel
```

#### `scale(T factor)`
Multiply all elements by factor.

```cpp
arr.to_device(NDARRAY_DEVICE_CUDA);
arr.scale(2.0f);  // GPU kernel: arr[i] *= 2.0
```

#### `add(const ndarray& other)`
Element-wise addition.

```cpp
arr1.to_device(NDARRAY_DEVICE_CUDA);
arr2.to_device(NDARRAY_DEVICE_CUDA);
arr1.add(arr2);  // GPU kernel: arr1[i] += arr2[i]
```

**Note**: Both arrays must be on same device type.

## Device Types

```cpp
enum {
  NDARRAY_DEVICE_HOST,   // CPU
  NDARRAY_DEVICE_CUDA,   // NVIDIA GPU (via CUDA)
  NDARRAY_DEVICE_HIP,    // AMD GPU (via HIP)
  NDARRAY_DEVICE_SYCL    // Cross-platform (Intel, NVIDIA, AMD)
};
```

## Memory Management (RAII)

GPU memory is **automatically managed** using RAII:

```cpp
{
  ndarray<float> arr({1000, 1000});
  arr.to_device(NDARRAY_DEVICE_CUDA);
  // GPU memory allocated here

  // ... use array ...

} // GPU memory automatically freed here (destructor)
```

**Benefits**:
- No manual `cudaFree()` required
- Exception-safe cleanup
- No memory leaks

## Multi-GPU Support

```cpp
// Use GPU 0
ndarray<float> arr1({1024, 1024});
arr1.to_device(NDARRAY_DEVICE_CUDA, 0);

// Use GPU 1
ndarray<float> arr2({1024, 1024});
arr2.to_device(NDARRAY_DEVICE_CUDA, 1);
```

## SYCL Support

For cross-platform GPU support:

```cpp
#include <sycl/sycl.hpp>

// Create SYCL queue
sycl::queue q(sycl::default_selector{});

ndarray<float> arr({1024, 1024});
arr.set_sycl_queue(&q);  // Optional: use specific queue
arr.to_device(NDARRAY_DEVICE_SYCL);
```

## Performance Tips

### 1. Minimize Transfers
```cpp
// Bad: Multiple transfers
for (int i = 0; i < 1000; i++) {
  arr.to_device(NDARRAY_DEVICE_CUDA);
  arr.scale(2.0f);
  arr.to_host();
}

// Good: Single transfer
arr.to_device(NDARRAY_DEVICE_CUDA);
for (int i = 0; i < 1000; i++) {
  arr.scale(2.0f);
}
arr.to_host();
```

### 2. Use copy_to_device() for Read-Only Data
```cpp
// Keep host copy for later use
arr.copy_to_device(NDARRAY_DEVICE_CUDA);
// ... GPU operations ...
arr.copy_from_device();
```

### 3. Batch Operations
```cpp
arr.to_device(NDARRAY_DEVICE_CUDA);
arr.fill(1.0f);
arr.scale(2.0f);
arr.add(other);  // All on GPU
arr.to_host();
```

## Testing

Run GPU tests:

```bash
# Build with CUDA
cmake -B build -DNDARRAY_USE_CUDA=TRUE -DNDARRAY_BUILD_TESTS=ON
cmake --build build

# Run tests
cd build
./bin/test_gpu          # Basic GPU tests
./bin/test_gpu_kernels  # Comprehensive data management tests
```

Test coverage:
- Device allocation/deallocation
- Host ↔ Device transfers (to_device, to_host)
- Bidirectional copy (copy_to_device, copy_from_device)
- GPU kernels (fill, scale, add)
- RAII cleanup verification
- Multiple round-trip transfers
- Large array transfers (4 MB+)

## Limitations

### What is NOT Supported

1. **Complex Compute Kernels**: No FFT, convolution, matrix multiply kernels
2. **Automatic Offloading**: You must explicitly call `to_device()`
3. **CPU/GPU Transparent Operations**: Operations fail if array not on expected device
4. **HIP/SYCL Production Use**: Only CUDA is production-ready
5. **Unified Memory**: No `cudaMallocManaged()` support

### Supported Data Types

- `float`, `double`
- `int`, `unsigned int`
- `char`, `unsigned char`

Custom types not supported for GPU kernels.

## Example: Complete Workflow

```cpp
#include <ndarray/ndarray.hh>
#include <iostream>

using namespace ftk;

int main() {
  // Create 2D array on host
  ndarray<float> temperature({1000, 1000});
  temperature.fill(273.15f);  // Initialize to 0°C in Kelvin

  std::cout << "Array on host, size: "
            << temperature.nelem() * sizeof(float) / 1e6
            << " MB" << std::endl;

  // Move to GPU
  temperature.to_device(NDARRAY_DEVICE_CUDA);
  std::cout << "Transferred to GPU" << std::endl;

  // GPU operations
  temperature.scale(1.8f);  // Scale

  // Move back to host
  temperature.to_host();
  std::cout << "Transferred back to host" << std::endl;

  return 0;
}
```

---

**Questions?** See main README.md or open an issue on GitHub.
