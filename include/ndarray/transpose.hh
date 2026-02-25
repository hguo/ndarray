#ifndef _NDARRAY_TRANSPOSE_HH
#define _NDARRAY_TRANSPOSE_HH

#include <ndarray/ndarray.hh>
#include <ndarray/error.hh>
#include <vector>
#include <algorithm>
#include <set>
#include <cstring>

#if NDARRAY_HAVE_MPI
#include <ndarray/transpose_distributed.hh>
#endif

namespace ftk {

// Tunable block size for cache-friendly transpose
// Optimized for typical L1 cache size (32-64KB)
// 64x64 doubles = 32KB, fits in L1 cache on most CPUs
constexpr size_t TRANSPOSE_BLOCK_SIZE = 64;

namespace detail {

/**
 * @brief Check if transpose permutation preserves component/time dimension semantics
 *
 * Returns true if the permutation keeps component dimensions at the beginning
 * and time dimension at the end (if present).
 *
 * @param axes Permutation axes
 * @param n_component_dims Number of leading component dimensions
 * @param has_time Whether last dimension is time
 * @return true if semantics are preserved, false otherwise
 */
inline bool preserves_dimension_semantics(const std::vector<size_t>& axes,
                                          size_t n_component_dims,
                                          bool has_time) {
  const size_t nd = axes.size();

  // Check if component dimensions stay at the beginning
  if (n_component_dims > 0) {
    for (size_t i = 0; i < n_component_dims; i++) {
      // Component dimensions should map to first n_component_dims positions
      if (axes[i] >= n_component_dims) {
        return false;  // Component dimension moved to non-component position
      }
    }
  }

  // Check if time dimension stays at the end
  if (has_time) {
    // Last axis should map to last dimension
    if (axes[nd - 1] != nd - 1) {
      return false;  // Time dimension was moved
    }
  }

  return true;
}

/**
 * @brief Validate transpose axes permutation
 * @param nd Number of dimensions
 * @param axes Permutation axes
 * @throws invalid_operation if axes are invalid
 */
inline void validate_transpose_axes(size_t nd, const std::vector<size_t>& axes) {
  if (axes.size() != nd) {
    throw invalid_operation("transpose: axes size (" + std::to_string(axes.size()) +
                       ") must match array dimensions (" + std::to_string(nd) + ")");
  }

  // Check for duplicate axes
  std::set<size_t> unique_axes(axes.begin(), axes.end());
  if (unique_axes.size() != axes.size()) {
    throw invalid_operation("transpose: axes must be unique (no duplicates allowed)");
  }

  // Check for out-of-range axes
  for (size_t i = 0; i < axes.size(); i++) {
    if (axes[i] >= nd) {
      throw invalid_operation("transpose: axis " + std::to_string(i) +
                         " has value " + std::to_string(axes[i]) +
                         " which is out of range [0, " + std::to_string(nd-1) + "]");
    }
  }
}

/**
 * @brief Blocked 2D transpose for cache efficiency
 *
 * Uses cache blocking to improve performance on large matrices.
 * Processes the matrix in BLOCKxBLOCK tiles to maximize cache reuse.
 *
 * @param input Input matrix
 * @param output Output matrix (must be pre-allocated with transposed dimensions)
 * @param block_size Block size for tiling
 */
template <typename T, typename StoragePolicy>
void transpose_2d_blocked(const ndarray<T, StoragePolicy>& input,
                          ndarray<T, StoragePolicy>& output,
                          size_t block_size = TRANSPOSE_BLOCK_SIZE) {
  const size_t rows = input.dimf(0);  // First dimension (Fortran order)
  const size_t cols = input.dimf(1);  // Second dimension

  // Process in blocks for better cache utilization
  for (size_t ii = 0; ii < rows; ii += block_size) {
    for (size_t jj = 0; jj < cols; jj += block_size) {
      // Compute actual block boundaries (handle partial blocks at edges)
      const size_t imax = std::min(ii + block_size, rows);
      const size_t jmax = std::min(jj + block_size, cols);

      // Transpose this block
      for (size_t i = ii; i < imax; i++) {
        for (size_t j = jj; j < jmax; j++) {
          output.f(j, i) = input.f(i, j);
        }
      }
    }
  }
}

/**
 * @brief In-place transpose for square 2D matrices
 *
 * Transposes a square matrix by swapping elements across the diagonal.
 * This is memory-efficient as it requires no additional storage.
 *
 * @param array Square matrix to transpose in-place
 */
template <typename T, typename StoragePolicy>
void transpose_2d_square_inplace(ndarray<T, StoragePolicy>& array) {
  const size_t n = array.dimf(0);

  // Verify it's actually square
  if (array.dimf(1) != n) {
    throw invalid_operation("transpose_inplace: matrix must be square (got " +
                       std::to_string(array.dimf(0)) + "x" + std::to_string(array.dimf(1)) + ")");
  }

  // Swap elements above and below diagonal
  // Only iterate over upper triangle to avoid double-swapping
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      std::swap(array.f(i, j), array.f(j, i));
    }
  }
}

/**
 * @brief General N-dimensional transpose
 *
 * Handles arbitrary axis permutations for N-dimensional arrays.
 * This is the most general but also slowest transpose implementation.
 *
 * @param input Input array
 * @param axes Permutation of axes
 * @return Transposed array
 */
template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> transpose_nd_general(const ndarray<T, StoragePolicy>& input,
                                                const std::vector<size_t>& axes) {
  const size_t nd = input.nd();
  validate_transpose_axes(nd, axes);

  // Compute output dimensions
  std::vector<size_t> output_dims(nd);
  for (size_t i = 0; i < nd; i++) {
    output_dims[i] = input.dimf(axes[i]);
  }

  // Create output array
  ndarray<T, StoragePolicy> output;
  output.reshapef(output_dims);

  // Handle metadata: Check if permutation preserves dimension semantics
  const size_t n_comp = input.multicomponents();
  const bool has_time = input.has_time();

  if (n_comp > 0 || has_time) {
    // Check if the permutation preserves component/time semantics
    if (!preserves_dimension_semantics(axes, n_comp, has_time)) {
      // Permutation moves component or time dimensions - warn user
      // We still copy the metadata, but it may no longer be semantically correct
      std::cerr << "[NDARRAY WARNING] transpose: permutation moves component or time dimensions." << std::endl;
      std::cerr << "  The resulting array may have incorrect metadata." << std::endl;
      std::cerr << "  Original: n_component_dims=" << n_comp << ", has_time=" << has_time << std::endl;
      std::cerr << "  Consider manually adjusting metadata after transpose." << std::endl;
    }
  }

  // Copy metadata (may be semantically incorrect after certain permutations)
  output.set_multicomponents(input.multicomponents());
  output.set_has_time(input.has_time());

  // Get input dimensions for index computation
  std::vector<size_t> input_dims(nd);
  for (size_t i = 0; i < nd; i++) {
    input_dims[i] = input.dimf(i);
  }

  // Permute data
  const size_t total_elements = input.nelem();

  if (total_elements == 0) {
    return output;  // Empty array
  }

  // For each output element, compute its input location
  std::vector<size_t> output_indices(nd);
  std::vector<size_t> input_indices(nd);

  for (size_t i = 0; i < total_elements; i++) {
    // Convert linear output index to multi-dimensional indices
    size_t idx = i;
    for (size_t d = 0; d < nd; d++) {
      output_indices[d] = idx % output_dims[d];
      idx /= output_dims[d];
    }

    // Apply inverse permutation to get input indices
    for (size_t d = 0; d < nd; d++) {
      input_indices[axes[d]] = output_indices[d];
    }

    // Copy element
    output[i] = input.f(input_indices);
  }

  return output;
}

} // namespace detail

/**
 * @brief Transpose array with specified axis permutation
 *
 * Reorders the dimensions of the input array according to the permutation
 * specified by `axes`. This is the most general transpose operation.
 *
 * @param input Input array
 * @param axes Permutation of axes (e.g., {2, 0, 1} for shape (a,b,c) → (c,a,b))
 * @return Transposed array with reordered dimensions
 *
 * @throws invalid_operation if axes is invalid (wrong size, duplicates, out of range)
 *
 * @example
 * ftk::ndarray<double> arr;
 * arr.reshapef(2, 3, 4);  // shape (2, 3, 4)
 * auto transposed = ftk::transpose(arr, {2, 0, 1});  // shape (4, 2, 3)
 */
template <typename T, typename StoragePolicy = native_storage>
ndarray<T, StoragePolicy> transpose(const ndarray<T, StoragePolicy>& input,
                                    const std::vector<size_t>& axes) {
  const size_t nd = input.nd();

  // Handle edge cases
  if (nd == 0) {
    // Scalar - return copy
    return input;
  }

  if (nd == 1) {
    // 1D array - transpose is identity
    return input;
  }

  detail::validate_transpose_axes(nd, axes);

#if NDARRAY_HAVE_MPI
  // Dispatch to distributed implementation if array is distributed
  if (input.is_distributed()) {
    return detail::transpose_distributed(input, axes);
  }
#endif

  // Check if this is actually an identity permutation
  bool is_identity = true;
  for (size_t i = 0; i < nd; i++) {
    if (axes[i] != i) {
      is_identity = false;
      break;
    }
  }
  if (is_identity) {
    return input;  // Return copy
  }

  // Special case for 2D transpose
  if (nd == 2 && axes[0] == 1 && axes[1] == 0) {
    ndarray<T, StoragePolicy> output;
    output.reshapef(input.dimf(1), input.dimf(0));

    // Copy metadata
    output.set_multicomponents(input.multicomponents());
    output.set_has_time(input.has_time());

    detail::transpose_2d_blocked(input, output);
    return output;
  }

  // General N-D transpose
  return detail::transpose_nd_general(input, axes);
}

/**
 * @brief Transpose 2D matrix (shorthand)
 *
 * Transposes a 2D matrix by swapping rows and columns.
 * This is equivalent to transpose(input, {1, 0}).
 *
 * @param input Input 2D matrix
 * @return Transposed matrix
 *
 * @throws invalid_operation if input is not 2D
 *
 * @example
 * ftk::ndarray<double> A;
 * A.reshapef(3, 4);  // 3x4 matrix
 * auto At = ftk::transpose(A);  // 4x3 matrix
 */
template <typename T, typename StoragePolicy = native_storage>
ndarray<T, StoragePolicy> transpose(const ndarray<T, StoragePolicy>& input) {
  if (input.nd() != 2) {
    throw invalid_operation("transpose() without axes requires 2D array (got " +
                       std::to_string(input.nd()) + "D)");
  }

  ndarray<T, StoragePolicy> output;
  output.reshapef(input.dimf(1), input.dimf(0));

  // Copy metadata
  output.set_multicomponents(input.multicomponents());
  output.set_has_time(input.has_time());

  detail::transpose_2d_blocked(input, output);
  return output;
}

/**
 * @brief In-place transpose for square 2D matrices
 *
 * Transposes a square matrix in-place, which saves memory by avoiding
 * allocation of a second array. This is ideal for large square matrices.
 *
 * @param array Square matrix to transpose (modified in-place)
 *
 * @throws invalid_operation if array is not 2D or not square
 *
 * @example
 * ftk::ndarray<double> A;
 * A.reshapef(100, 100);  // Square matrix
 * ftk::transpose_inplace(A);  // Transposed in-place
 */
template <typename T, typename StoragePolicy = native_storage>
void transpose_inplace(ndarray<T, StoragePolicy>& array) {
  if (array.nd() != 2) {
    throw invalid_operation("transpose_inplace: requires 2D array (got " +
                       std::to_string(array.nd()) + "D)");
  }

  if (array.dimf(0) != array.dimf(1)) {
    throw invalid_operation("transpose_inplace: requires square matrix (got " +
                       std::to_string(array.dimf(0)) + "x" + std::to_string(array.dimf(1)) + ")");
  }

  detail::transpose_2d_square_inplace(array);
}

} // namespace ftk

#endif // _NDARRAY_TRANSPOSE_HH
