#ifndef _NDARRAY_TRANSPOSE_CUDA_HH
#define _NDARRAY_TRANSPOSE_CUDA_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_CUDA

#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_cuda.hh>
#include <ndarray/error.hh>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace ftk {
namespace detail {

#ifdef __CUDACC__
// Only compile CUDA device code when using nvcc compiler

// Tile size for shared memory transpose (32x32 is optimal for most GPUs)
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;  // To avoid shared memory bank conflicts

/**
 * @brief CUDA kernel for 2D matrix transpose using shared memory
 *
 * Optimized transpose kernel that uses shared memory tiles to achieve
 * coalesced memory access patterns. This is much faster than naive transpose.
 *
 * Key optimizations:
 * - 32×32 shared memory tiles for cache blocking
 * - Coalesced reads from input (transpose happens in shared memory)
 * - Coalesced writes to output (with bank conflict avoidance)
 *
 * @param input Input matrix (row-major layout in device memory)
 * @param output Output matrix (transposed, row-major layout)
 * @param width Width of input (number of columns)
 * @param height Height of input (number of rows)
 */
template <typename T>
__global__ void transpose_2d_kernel(const T* __restrict__ input,
                                    T* __restrict__ output,
                                    int width, int height) {
  // Shared memory tile with padding to avoid bank conflicts
  __shared__ T tile[TILE_DIM][TILE_DIM + 1];

  // Global input coordinates
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  // Read tile from input (coalesced access)
  // Each thread reads BLOCK_ROWS elements to improve occupancy
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    int row = y + j;
    if (x < width && row < height) {
      tile[threadIdx.y + j][threadIdx.x] = input[row * width + x];
    }
  }

  __syncthreads();

  // Global output coordinates (transposed)
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  // Write tile to output (coalesced access)
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    int row = y + j;
    if (x < height && row < width) {
      output[row * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

/**
 * @brief CUDA kernel for N-dimensional transpose (general case)
 *
 * This kernel handles arbitrary dimension permutations by computing
 * the mapping from output index to input index.
 *
 * Less efficient than specialized 2D kernel but works for any ndim.
 *
 * @param input Input array (fortran-order layout)
 * @param output Output array (fortran-order layout with permuted dims)
 * @param n_elems Total number of elements
 * @param ndim Number of dimensions
 * @param input_dims Input dimensions (fortran order)
 * @param output_dims Output dimensions (fortran order)
 * @param axes Permutation axes
 */
template <typename T>
__global__ void transpose_nd_kernel(const T* __restrict__ input,
                                    T* __restrict__ output,
                                    size_t n_elems,
                                    int ndim,
                                    const size_t* __restrict__ input_dims,
                                    const size_t* __restrict__ output_dims,
                                    const size_t* __restrict__ axes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n_elems) return;

  // Compute multi-dimensional output index from linear index
  size_t tmp = idx;
  size_t output_idx[16];  // Max 16 dimensions
  for (int d = 0; d < ndim; d++) {
    output_idx[d] = tmp % output_dims[d];
    tmp /= output_dims[d];
  }

  // Apply inverse permutation to get input index
  size_t input_idx[16];
  for (int d = 0; d < ndim; d++) {
    input_idx[axes[d]] = output_idx[d];
  }

  // Compute linear input index from multi-dimensional index
  size_t input_linear = 0;
  size_t stride = 1;
  for (int d = 0; d < ndim; d++) {
    input_linear += input_idx[d] * stride;
    stride *= input_dims[d];
  }

  // Copy element
  output[idx] = input[input_linear];
}

#endif // __CUDACC__

/**
 * @brief Host function: 2D transpose on CUDA device
 */
template <typename T, typename StoragePolicy>
void transpose_2d_cuda(const ndarray<T, StoragePolicy>& input,
                       ndarray<T, StoragePolicy>& output) {
#ifdef __CUDACC__
  // ndarray stores data in column-major (Fortran) order internally.
  // From the kernel's row-major perspective, the raw memory is a
  // dimf(1) x dimf(0) matrix (height=cols, width=rows).
  const size_t width = input.dimf(0);
  const size_t height = input.dimf(1);

  // Ensure input is on device
  if (!input.is_on_device() || input.get_device_type() != NDARRAY_DEVICE_CUDA) {
    throw device_error(ERR_NOT_BUILT_WITH_CUDA,
                      "transpose_2d_cuda: input array is not on CUDA device");
  }

  // Ensure output is on same device
  if (!output.is_on_device() || output.get_device_type() != NDARRAY_DEVICE_CUDA) {
    output.to_device(NDARRAY_DEVICE_CUDA, input.get_device_id());
  }

  const T* d_input = static_cast<const T*>(input.get_devptr());
  T* d_output = static_cast<T*>(output.get_devptr());

  // Launch kernel with optimal grid/block configuration
  dim3 blockDim(TILE_DIM, BLOCK_ROWS);
  dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
               (height + TILE_DIM - 1) / TILE_DIM);

  transpose_2d_kernel<T><<<gridDim, blockDim>>>(
      d_input, d_output, width, height);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
#else
  throw device_error(ERR_NOT_BUILT_WITH_CUDA,
                    "transpose_2d_cuda: CUDA support not available (not compiled with nvcc)");
#endif
}

/**
 * @brief Host function: N-D transpose on CUDA device
 */
template <typename T, typename StoragePolicy>
void transpose_nd_cuda(const ndarray<T, StoragePolicy>& input,
                       ndarray<T, StoragePolicy>& output,
                       const std::vector<size_t>& axes) {
#ifdef __CUDACC__
  const size_t nd = input.nd();
  const size_t n_elems = input.nelem();

  // Ensure input is on device
  if (!input.is_on_device() || input.get_device_type() != NDARRAY_DEVICE_CUDA) {
    throw device_error(ERR_NOT_BUILT_WITH_CUDA,
                      "transpose_nd_cuda: input array is not on CUDA device");
  }

  // Ensure output is on same device
  if (!output.is_on_device() || output.get_device_type() != NDARRAY_DEVICE_CUDA) {
    output.to_device(NDARRAY_DEVICE_CUDA, input.get_device_id());
  }

  // Get input/output dimensions
  std::vector<size_t> input_dims(nd);
  std::vector<size_t> output_dims(nd);
  for (size_t i = 0; i < nd; i++) {
    input_dims[i] = input.dimf(i);
    output_dims[i] = output.dimf(i);
  }

  // Allocate device memory for dimension arrays
  size_t* d_input_dims;
  size_t* d_output_dims;
  size_t* d_axes;

  CUDA_CHECK(cudaMalloc(&d_input_dims, nd * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_output_dims, nd * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_axes, nd * sizeof(size_t)));

  CUDA_CHECK(cudaMemcpy(d_input_dims, input_dims.data(),
                       nd * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_output_dims, output_dims.data(),
                       nd * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_axes, axes.data(),
                       nd * sizeof(size_t), cudaMemcpyHostToDevice));

  const T* d_input = static_cast<const T*>(input.get_devptr());
  T* d_output = static_cast<T*>(output.get_devptr());

  // Launch kernel
  const int blockSize = 256;
  const int gridSize = (n_elems + blockSize - 1) / blockSize;

  transpose_nd_kernel<T><<<gridSize, blockSize>>>(
      d_input, d_output, n_elems, nd,
      d_input_dims, d_output_dims, d_axes);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Free device memory
  CUDA_CHECK(cudaFree(d_input_dims));
  CUDA_CHECK(cudaFree(d_output_dims));
  CUDA_CHECK(cudaFree(d_axes));
#else
  throw device_error(ERR_NOT_BUILT_WITH_CUDA,
                    "transpose_nd_cuda: CUDA support not available (not compiled with nvcc)");
#endif
}

/**
 * @brief Main CUDA transpose dispatcher
 *
 * Dispatches to optimized 2D kernel for simple 2D transpose,
 * or general N-D kernel for arbitrary permutations.
 *
 * @param input Input array on CUDA device
 * @param axes Permutation axes
 * @return Transposed array on CUDA device
 */
template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> transpose_cuda(const ndarray<T, StoragePolicy>& input,
                                         const std::vector<size_t>& axes) {
  const size_t nd = input.nd();

  // Compute output dimensions
  std::vector<size_t> output_dims(nd);
  for (size_t i = 0; i < nd; i++) {
    output_dims[i] = input.dimf(axes[i]);
  }

  // Create output array
  ndarray<T, StoragePolicy> output;
  output.reshapef(output_dims);

  // Copy metadata
  output.set_multicomponents(input.multicomponents());
  output.set_has_time(input.has_time());

  // Move output to device
  output.to_device(NDARRAY_DEVICE_CUDA, input.get_device_id());

  // Dispatch to appropriate kernel
  if (nd == 2 && axes[0] == 1 && axes[1] == 0) {
    // Simple 2D transpose - use optimized kernel
    transpose_2d_cuda(input, output);
  } else {
    // General N-D transpose
    transpose_nd_cuda(input, output, axes);
  }

  return output;
}

} // namespace detail
} // namespace ftk

#endif // NDARRAY_HAVE_CUDA
#endif // _NDARRAY_TRANSPOSE_CUDA_HH
