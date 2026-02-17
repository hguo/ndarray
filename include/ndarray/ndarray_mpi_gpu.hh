#ifndef _NDARRAY_NDARRAY_MPI_GPU_HH
#define _NDARRAY_NDARRAY_MPI_GPU_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_CUDA && NDARRAY_HAVE_MPI

#include <cuda_runtime.h>

namespace ftk {
namespace nd {

// Only include CUDA kernel implementations when compiling with nvcc
// __CUDACC__ is defined by nvcc, not by regular C++ compilers
#ifdef __CUDACC__

//==============================================================================
// 1D Kernels
//==============================================================================

/**
 * CUDA kernel to pack boundary data from 1D device array
 */
template <typename T>
__global__ void pack_boundary_1d_kernel(
    T* buffer,
    const T* array,
    int dim0,
    bool is_high,
    int ghost_width,
    int core_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= ghost_width) return;

  int src_i = is_high ? (core_size - ghost_width + idx) : idx;
  buffer[idx] = array[src_i];
}

/**
 * CUDA kernel to unpack received ghost data into 1D device array
 */
template <typename T>
__global__ void unpack_ghost_1d_kernel(
    T* array,
    const T* buffer,
    int dim0,
    bool is_high,
    int ghost_width,
    int ghost_low,
    int ghost_high,
    int core_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= ghost_width) return;

  int dst_i;
  if (!is_high && ghost_low > 0) {
    // Left ghost
    dst_i = idx;
  } else if (is_high && ghost_high > 0) {
    // Right ghost
    dst_i = core_size + ghost_low + idx;
  } else {
    return;  // No ghost on this side
  }

  array[dst_i] = buffer[idx];
}

//==============================================================================
// 2D Kernels
//==============================================================================

/**
 * CUDA kernel to pack boundary data from device array
 *
 * Copies boundary slice from full array into contiguous buffer
 * for MPI communication.
 *
 * @param buffer Output buffer (device memory)
 * @param array Input array (device memory)
 * @param dim0, dim1 Array dimensions
 * @param boundary_dim Which dimension the boundary is in (0 or 1)
 * @param is_high Whether this is high boundary (true) or low (false)
 * @param ghost_width Width of ghost layer
 * @param core_size0, core_size1 Core region sizes
 */
template <typename T>
__global__ void pack_boundary_2d_kernel(
    T* buffer,
    const T* array,
    int dim0, int dim1,
    int boundary_dim,
    bool is_high,
    int ghost_width,
    int core_size0, int core_size1)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int total = (boundary_dim == 0) ? (ghost_width * core_size1)
                                   : (core_size0 * ghost_width);

  if (idx >= total) return;

  if (boundary_dim == 0) {
    // Boundary in dimension 0 (left/right)
    int i = idx / core_size1;
    int j = idx % core_size1;

    int src_i = is_high ? (core_size0 - ghost_width + i) : i;
    buffer[idx] = array[src_i * dim1 + j];
  } else {
    // Boundary in dimension 1 (up/down)
    int i = idx / ghost_width;
    int j = idx % ghost_width;

    int src_j = is_high ? (core_size1 - ghost_width + j) : j;
    buffer[idx] = array[i * dim1 + src_j];
  }
}

/**
 * CUDA kernel to unpack received ghost data into device array
 *
 * Copies received ghost data from contiguous buffer into ghost
 * regions of the array.
 *
 * @param array Output array (device memory)
 * @param buffer Input buffer with received data (device memory)
 * @param dim0, dim1 Array dimensions
 * @param ghost_dim Which dimension the ghost is in (0 or 1)
 * @param is_high Whether this is high ghost (true) or low (false)
 * @param ghost_width Width of ghost layer
 * @param ghost_low, ghost_high Ghost offsets
 * @param core_size0, core_size1 Core region sizes
 */
template <typename T>
__global__ void unpack_ghost_2d_kernel(
    T* array,
    const T* buffer,
    int dim0, int dim1,
    int ghost_dim,
    bool is_high,
    int ghost_width,
    int ghost_low,
    int ghost_high,
    int core_size0, int core_size1)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int total = (ghost_dim == 0) ? (ghost_width * core_size1)
                                : (core_size0 * ghost_width);

  if (idx >= total) return;

  if (ghost_dim == 0) {
    // Ghost in dimension 0
    int i = idx / core_size1;
    int j = idx % core_size1;

    int dst_i;
    if (!is_high && ghost_low > 0) {
      // Left ghost
      dst_i = i;
    } else if (is_high && ghost_high > 0) {
      // Right ghost
      dst_i = core_size0 + ghost_low + i;
    } else {
      return;  // No ghost on this side
    }

    array[dst_i * dim1 + j] = buffer[idx];
  } else {
    // Ghost in dimension 1
    int i = idx / ghost_width;
    int j = idx % ghost_width;

    int dst_j;
    if (!is_high && ghost_low > 0) {
      // Bottom ghost
      dst_j = j;
    } else if (is_high && ghost_high > 0) {
      // Top ghost
      dst_j = core_size1 + ghost_low + j;
    } else {
      return;
    }

    array[i * dim1 + dst_j] = buffer[idx];
  }
}

//==============================================================================
// 3D Kernels
//==============================================================================

/**
 * CUDA kernel to pack boundary data from 3D device array
 */
template <typename T>
__global__ void pack_boundary_3d_kernel(
    T* buffer,
    const T* array,
    int dim0, int dim1, int dim2,
    int boundary_dim,
    bool is_high,
    int ghost_width,
    int core_size0, int core_size1, int core_size2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int total;
  if (boundary_dim == 0) {
    total = ghost_width * core_size1 * core_size2;
  } else if (boundary_dim == 1) {
    total = core_size0 * ghost_width * core_size2;
  } else {
    total = core_size0 * core_size1 * ghost_width;
  }

  if (idx >= total) return;

  int src_i, src_j, src_k;

  if (boundary_dim == 0) {
    // Boundary in dimension 0
    int i = idx / (core_size1 * core_size2);
    int j = (idx % (core_size1 * core_size2)) / core_size2;
    int k = idx % core_size2;

    src_i = is_high ? (core_size0 - ghost_width + i) : i;
    src_j = j;
    src_k = k;
  } else if (boundary_dim == 1) {
    // Boundary in dimension 1
    int i = idx / (ghost_width * core_size2);
    int j = (idx % (ghost_width * core_size2)) / core_size2;
    int k = idx % core_size2;

    src_i = i;
    src_j = is_high ? (core_size1 - ghost_width + j) : j;
    src_k = k;
  } else {
    // Boundary in dimension 2
    int i = idx / (core_size1 * ghost_width);
    int j = (idx % (core_size1 * ghost_width)) / ghost_width;
    int k = idx % ghost_width;

    src_i = i;
    src_j = j;
    src_k = is_high ? (core_size2 - ghost_width + k) : k;
  }

  buffer[idx] = array[src_i * dim1 * dim2 + src_j * dim2 + src_k];
}

/**
 * CUDA kernel to unpack received ghost data into 3D device array
 */
template <typename T>
__global__ void unpack_ghost_3d_kernel(
    T* array,
    const T* buffer,
    int dim0, int dim1, int dim2,
    int ghost_dim,
    bool is_high,
    int ghost_width,
    int ghost_low,
    int ghost_high,
    int core_size0, int core_size1, int core_size2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int total;
  if (ghost_dim == 0) {
    total = ghost_width * core_size1 * core_size2;
  } else if (ghost_dim == 1) {
    total = core_size0 * ghost_width * core_size2;
  } else {
    total = core_size0 * core_size1 * ghost_width;
  }

  if (idx >= total) return;

  int dst_i, dst_j, dst_k;

  if (ghost_dim == 0) {
    // Ghost in dimension 0
    int i = idx / (core_size1 * core_size2);
    int j = (idx % (core_size1 * core_size2)) / core_size2;
    int k = idx % core_size2;

    if (!is_high && ghost_low > 0) {
      dst_i = i;
    } else if (is_high && ghost_high > 0) {
      dst_i = core_size0 + ghost_low + i;
    } else {
      return;
    }
    dst_j = j;
    dst_k = k;

  } else if (ghost_dim == 1) {
    // Ghost in dimension 1
    int i = idx / (ghost_width * core_size2);
    int j = (idx % (ghost_width * core_size2)) / core_size2;
    int k = idx % core_size2;

    dst_i = i;
    if (!is_high && ghost_low > 0) {
      dst_j = j;
    } else if (is_high && ghost_high > 0) {
      dst_j = core_size1 + ghost_low + j;
    } else {
      return;
    }
    dst_k = k;

  } else {
    // Ghost in dimension 2
    int i = idx / (core_size1 * ghost_width);
    int j = (idx % (core_size1 * ghost_width)) / ghost_width;
    int k = idx % ghost_width;

    dst_i = i;
    dst_j = j;
    if (!is_high && ghost_low > 0) {
      dst_k = k;
    } else if (is_high && ghost_high > 0) {
      dst_k = core_size2 + ghost_low + k;
    } else {
      return;
    }
  }

  array[dst_i * dim1 * dim2 + dst_j * dim2 + dst_k] = buffer[idx];
}

//==============================================================================
// Launcher Functions
//==============================================================================

/**
 * Launch pack_boundary kernel for 1D arrays
 */
template <typename T>
inline void launch_pack_boundary_1d(
    T* d_buffer,
    const T* d_array,
    int dim0,
    bool is_high,
    int ghost_width,
    int core_size,
    cudaStream_t stream = 0)
{
  int total = ghost_width;
  int threads_per_block = 256;
  int num_blocks = (total + threads_per_block - 1) / threads_per_block;

  pack_boundary_1d_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_buffer, d_array, dim0, is_high, ghost_width, core_size);
}

/**
 * Launch unpack_ghost kernel for 1D arrays
 */
template <typename T>
inline void launch_unpack_ghost_1d(
    T* d_array,
    const T* d_buffer,
    int dim0,
    bool is_high,
    int ghost_width,
    int ghost_low,
    int ghost_high,
    int core_size,
    cudaStream_t stream = 0)
{
  int total = ghost_width;
  int threads_per_block = 256;
  int num_blocks = (total + threads_per_block - 1) / threads_per_block;

  unpack_ghost_1d_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_array, d_buffer, dim0, is_high, ghost_width,
      ghost_low, ghost_high, core_size);
}

/**
 * Launch pack_boundary kernel with appropriate grid/block configuration
 */
template <typename T>
inline void launch_pack_boundary_2d(
    T* d_buffer,
    const T* d_array,
    int dim0, int dim1,
    int boundary_dim,
    bool is_high,
    int ghost_width,
    int core_size0, int core_size1,
    cudaStream_t stream = 0)
{
  int total = (boundary_dim == 0) ? (ghost_width * core_size1)
                                   : (core_size0 * ghost_width);

  // Use 256 threads per block (typical optimal value)
  int threads_per_block = 256;
  int num_blocks = (total + threads_per_block - 1) / threads_per_block;

  pack_boundary_2d_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_buffer, d_array, dim0, dim1, boundary_dim, is_high,
      ghost_width, core_size0, core_size1);
}

/**
 * Launch unpack_ghost kernel with appropriate grid/block configuration
 */
template <typename T>
inline void launch_unpack_ghost_2d(
    T* d_array,
    const T* d_buffer,
    int dim0, int dim1,
    int ghost_dim,
    bool is_high,
    int ghost_width,
    int ghost_low,
    int ghost_high,
    int core_size0, int core_size1,
    cudaStream_t stream = 0)
{
  int total = (ghost_dim == 0) ? (ghost_width * core_size1)
                                : (core_size0 * ghost_width);

  int threads_per_block = 256;
  int num_blocks = (total + threads_per_block - 1) / threads_per_block;

  unpack_ghost_2d_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_array, d_buffer, dim0, dim1, ghost_dim, is_high,
      ghost_width, ghost_low, ghost_high, core_size0, core_size1);
}

/**
 * Launch pack_boundary kernel for 3D arrays
 */
template <typename T>
inline void launch_pack_boundary_3d(
    T* d_buffer,
    const T* d_array,
    int dim0, int dim1, int dim2,
    int boundary_dim,
    bool is_high,
    int ghost_width,
    int core_size0, int core_size1, int core_size2,
    cudaStream_t stream = 0)
{
  int total;
  if (boundary_dim == 0) {
    total = ghost_width * core_size1 * core_size2;
  } else if (boundary_dim == 1) {
    total = core_size0 * ghost_width * core_size2;
  } else {
    total = core_size0 * core_size1 * ghost_width;
  }

  int threads_per_block = 256;
  int num_blocks = (total + threads_per_block - 1) / threads_per_block;

  pack_boundary_3d_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_buffer, d_array, dim0, dim1, dim2, boundary_dim, is_high,
      ghost_width, core_size0, core_size1, core_size2);
}

/**
 * Launch unpack_ghost kernel for 3D arrays
 */
template <typename T>
inline void launch_unpack_ghost_3d(
    T* d_array,
    const T* d_buffer,
    int dim0, int dim1, int dim2,
    int ghost_dim,
    bool is_high,
    int ghost_width,
    int ghost_low,
    int ghost_high,
    int core_size0, int core_size1, int core_size2,
    cudaStream_t stream = 0)
{
  int total;
  if (ghost_dim == 0) {
    total = ghost_width * core_size1 * core_size2;
  } else if (ghost_dim == 1) {
    total = core_size0 * ghost_width * core_size2;
  } else {
    total = core_size0 * core_size1 * ghost_width;
  }

  int threads_per_block = 256;
  int num_blocks = (total + threads_per_block - 1) / threads_per_block;

  unpack_ghost_3d_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_array, d_buffer, dim0, dim1, dim2, ghost_dim, is_high,
      ghost_width, ghost_low, ghost_high, core_size0, core_size1, core_size2);
}

#endif // __CUDACC__

} // namespace nd
} // namespace ftk

#endif // NDARRAY_HAVE_CUDA && NDARRAY_HAVE_MPI

#endif // _NDARRAY_NDARRAY_MPI_GPU_HH
