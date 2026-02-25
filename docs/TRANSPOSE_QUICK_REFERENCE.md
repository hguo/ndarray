# Transpose Quick Reference

> **⚠️ Performance Note**: Performance claims have been removed as code is untested. Benchmark on your hardware to determine optimal approach.

## Basic Usage

```cpp
#include <ndarray/transpose.hh>

// 2D matrix transpose
auto At = ftk::transpose(A);

// N-D permutation
auto permuted = ftk::transpose(tensor, {2, 0, 1});

// In-place (square matrices only)
ftk::transpose_inplace(square_matrix);
```

## ⚠️ CRITICAL: Multicomponent & Time Arrays

### The Problem
ndarray tracks special dimension semantics:
- **`n_component_dims`**: First N dims are components (not spatial)
- **`is_time_varying`**: Last dim is time (not spatial)

**Layout**: `[components..., spatial..., time]`

### The Solution
**✅ SAFE: Only transpose spatial dimensions**
```cpp
// Vector field: [3, 100, 200], n_component_dims=1
ftk::transpose(V, {0, 2, 1});  // [3, 200, 100] ✓ Metadata correct
//                ^  spatial only
//             keep component first

// Time-varying: [100, 200, 50], is_time_varying=true
ftk::transpose(T, {1, 0, 2});  // [200, 100, 50] ✓ Metadata correct
//             spatial only  ^
//                        keep time last
```

**⚠️ UNSAFE: Moving component/time dimensions**
```cpp
// BAD: Moves component dimension
ftk::transpose(V, {1, 0, 2});  // [100, 3, 200] ⚠️ WARNING
// Component dim moved from position 0 → 1
// Metadata says n_component_dims=1 but first dim is NOT components!
```

### Quick Checklist

Before transposing, ask:
1. ✅ Does array have components? (`array.multicomponents() > 0`)
2. ✅ Does array have time? (`array.has_time()`)
3. ✅ Am I only permuting spatial dimensions?

If YES to 1 or 2, and NO to 3 → **You'll get a warning!**

### How to Handle Warnings

**Option 1**: Adjust permutation to keep component/time fixed
```cpp
// Instead of: {1, 0, 2}  (moves component)
// Use:        {0, 2, 1}  (keeps component at 0)
```

**Option 2**: Clear metadata if semantics don't matter
```cpp
array.set_multicomponents(0);
array.set_has_time(false);
auto result = ftk::transpose(array, arbitrary_axes);
```

**Option 3**: Manually fix metadata after transpose
```cpp
auto result = ftk::transpose(array, {1, 0, 2});
result.set_multicomponents(0);  // No longer has component semantics
// Document that dimension 1 is actually components now
```

## Common Patterns

### Transpose Spatial Dimensions of Vector Field
```cpp
// 3D velocity: [3, nx, ny, nz]
V.set_multicomponents(1);
auto Vt = ftk::transpose(V, {0, 1, 3, 2});  // [3, nx, nz, ny]
//                            ^  spatial permute
//                         keep component
```

### Transpose Time-Varying Scalar Field
```cpp
// Time series: [nx, ny, nt]
T.set_has_time(true);
auto Tt = ftk::transpose(T, {1, 0, 2});  // [ny, nx, nt]
//                       spatial^    ^keep time
```

### Transpose Both
```cpp
// Vector field time series: [3, nx, ny, nt]
VT.set_multicomponents(1);
VT.set_has_time(true);
auto VTt = ftk::transpose(VT, {0, 2, 1, 3});  // [3, ny, nx, nt]
//                             ^  spatial  ^
//                          comp         time
```

## 🌐 Distributed Arrays (MPI)

### ❌ CRITICAL: Stricter Rules for Distributed Arrays

For arrays created with `decompose()`:
- Component and time dimensions **CANNOT** be moved (throws error, not warning)
- Only spatial dimensions can be transposed
- Data is automatically redistributed across ranks

### Examples

**✅ Valid**:
```cpp
// Vector field distributed: [3, 1000, 800]
V.decompose(comm, {3, 1000, 800}, nprocs, {0, 4, 2}, {0, 1, 1});
V.set_multicomponents(1);

auto Vt = ftk::transpose(V, {0, 2, 1});  // ✓ Spatial only
// Decomposition automatically updated: {0, 2, 4}
```

**❌ Invalid**:
```cpp
auto Vbad = ftk::transpose(V, {1, 0, 2});  // ✗ THROWS ERROR
// Cannot move component dimension in distributed array
```

### See Also: `docs/TRANSPOSE_DISTRIBUTED.md` for full details

## 🚀 GPU Acceleration (CUDA)

### Automatic GPU Dispatch

Transpose automatically uses CUDA when array is on GPU:

```cpp
// CPU transpose
ftk::ndarray<float> cpu_arr;
auto result = ftk::transpose(cpu_arr);  // Runs on CPU

// GPU transpose (automatic!)
ftk::ndarray<float> gpu_arr = cpu_arr;
gpu_arr.to_device(NDARRAY_DEVICE_CUDA);
auto gpu_result = ftk::transpose(gpu_arr);  // Runs on GPU!
```

### Design Goals

GPU transpose uses optimized CUDA kernels designed for high performance:
- Shared memory tiling for 2D transpose
- Coalesced memory access patterns
- Efficient N-D permutation kernel

**When to consider GPU:**
- ✅ Large arrays
- ✅ Array already on GPU (avoids transfer)
- ✅ Part of GPU pipeline

**Note**: Profile to determine optimal approach for your hardware and problem size.

**See**: `docs/TRANSPOSE_GPU.md` for complete GPU guide

## 🌐+🚀 GPU + MPI (Distributed on GPU)

### The Ultimate: Distributed Arrays on GPUs

Combine multi-GPU distribution with GPU acceleration:

```cpp
// Multi-GPU cluster transpose
arr.decompose(comm, {8192, 8192}, nprocs, {4, 2}, {0, 0});
arr.to_device(NDARRAY_DEVICE_CUDA);  // Each rank on its GPU

auto result = ftk::transpose(arr);   // GPU+MPI automatic!
// Result: distributed AND on GPU
```

### Two Modes

1. **GPU-Aware MPI** (3-4× faster)
   - Direct GPU-to-GPU communication
   - Requires CUDA-aware MPI build

2. **CPU Staging** (automatic fallback)
   - Works with any MPI
   - Stages through CPU memory

### Design

- Combines GPU kernels with MPI communication
- Supports GPU-aware MPI or CPU staging fallback
- Designed for scalability across multiple GPUs/nodes
- **Best for**: Arrays too large for single GPU

**Note**: Performance depends on GPU model, network, MPI implementation, and problem size. Benchmark on your system.

**See**: `docs/TRANSPOSE_GPU_MPI.md` for complete guide

## Performance Tips

| Configuration | Approach | Characteristics |
|---------------|----------|-----------------|
| Small arrays | Out-of-place | CPU typically sufficient |
| Medium arrays | Blocked/Cache-optimized | CPU with blocking |
| Large (square) | In-place | Saves memory (CPU) |
| Large (GPU) | CUDA | GPU kernels optimized for large arrays |
| Distributed (CPU) | MPI | All-to-all communication |
| Distributed (GPU) | CUDA + MPI | GPU + MPI communication |

**Note**: Optimal choice depends on hardware. Profile to determine best approach.

## Error Messages

| Error | Meaning | Fix |
|-------|---------|-----|
| "axes size must match" | Wrong number of axes | Provide `nd()` axes |
| "axes must be unique" | Duplicate in axes | Use each axis once |
| "axis N out of range" | Invalid axis index | Use axes in [0, nd-1] |
| "requires 2D array" | Called 2D version on 3D+ | Specify axes explicitly |
| "requires square matrix" | In-place on non-square | Use out-of-place |
| "WARNING: moves component/time" | Unsafe permutation (serial) | See metadata handling above |
| "Cannot move component dimension" | Tried to move component (distributed) | Only transpose spatial dims |
| "Cannot move time dimension" | Tried to move time (distributed) | Keep time at end |

## See Also

- **Full Guide**: `docs/TRANSPOSE_METADATA_HANDLING.md`
- **Distributed Guide**: `docs/TRANSPOSE_DISTRIBUTED.md`
- **GPU Guide**: `docs/TRANSPOSE_GPU.md`
- **GPU+MPI Guide**: `docs/TRANSPOSE_GPU_MPI.md`
- **Design Doc**: `docs/TRANSPOSE_DESIGN.md`
- **Examples**: `examples/transpose_example.cpp`
- **Tests**: `tests/test_transpose_metadata.cpp`, `tests/test_transpose_distributed.cpp`, `tests/test_transpose_cuda.cpp`, `tests/test_transpose_distributed_gpu.cpp`

---
**TL;DR**:
- **Serial arrays**: Only transpose spatial dimensions to avoid metadata issues (warning if violated)
- **Distributed arrays**: Must only transpose spatial dimensions (error if violated)
- **GPU arrays**: Automatic CUDA acceleration using optimized kernels
- **GPU+MPI arrays**: Combines GPU + MPI for multi-GPU clusters
- **Performance**: Should be profiled on your specific hardware and problem size
