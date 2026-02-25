# Transpose API Design Document

## Overview

This document describes the design and implementation of fast transpose operations for the ndarray library. Transpose is a fundamental operation for multi-dimensional arrays, used extensively in linear algebra, tensor operations, and data layout transformations.

## Requirements

### Functional Requirements
1. **General Permutation**: Support arbitrary axis permutations for N-dimensional arrays
2. **2D Optimization**: Fast path for common 2D matrix transpose
3. **In-Place Option**: Memory-efficient in-place transpose for square matrices
4. **Multiple Backends**: Work with all storage policies (native, xtensor, eigen)
5. **Type Safety**: Maintain type safety across all operations
6. **Edge Cases**: Handle 0D, 1D, empty arrays correctly

### Performance Requirements
1. **Cache-Friendly**: Use blocked algorithms to maximize cache utilization
2. **Small Arrays**: Minimize overhead for small arrays (< 1KB)
3. **Large Arrays**: Scale efficiently to GB-sized arrays
4. **Parallelization**: Enable future SIMD/multi-threading optimizations

## API Design

### Primary APIs

```cpp
// 1. Out-of-place transpose with axis permutation
template <typename T, typename StoragePolicy = native_storage>
ndarray<T, StoragePolicy> transpose(const ndarray<T, StoragePolicy>& input,
                                    const std::vector<size_t>& axes);

// 2. 2D matrix transpose (shorthand for common case)
template <typename T, typename StoragePolicy = native_storage>
ndarray<T, StoragePolicy> transpose(const ndarray<T, StoragePolicy>& input);

// 3. In-place transpose for square 2D matrices
template <typename T, typename StoragePolicy = native_storage>
void transpose_inplace(ndarray<T, StoragePolicy>& input);

// 4. Member function versions
template <typename T, typename StoragePolicy>
struct ndarray {
    // Out-of-place transpose with permutation
    ndarray<T, StoragePolicy> transpose(const std::vector<size_t>& axes) const;

    // 2D transpose (shorthand)
    ndarray<T, StoragePolicy> transpose() const;
    ndarray<T, StoragePolicy> T() const;  // NumPy-style alias

    // In-place transpose (2D square only)
    void transpose_inplace();
};
```

### Usage Examples

```cpp
// Example 1: 2D matrix transpose
ftk::ndarray<double> A;
A.reshapef(3, 4);  // 3x4 matrix
// ... fill A ...
auto At = A.transpose();  // 4x3 matrix

// Example 2: 3D tensor permutation
ftk::ndarray<float> tensor;
tensor.reshapef(2, 3, 4);  // shape (2, 3, 4)
auto permuted = tensor.transpose({2, 0, 1});  // shape (4, 2, 3)

// Example 3: In-place square matrix transpose
ftk::ndarray<double> square;
square.reshapef(100, 100);
// ... fill square ...
square.transpose_inplace();  // Modifies in-place, saves memory

// Example 4: Using T() alias
auto At = A.T();  // Same as A.transpose()

// Example 5: Higher dimensions
ftk::ndarray<double> data4d;
data4d.reshapef(10, 20, 30, 40);
auto reordered = data4d.transpose({3, 1, 0, 2});  // Arbitrary permutation
```

## Implementation Strategy

### 1. Dimension Validation

```cpp
// Validate axes permutation
- Check axes vector length matches ndarray dimensions
- Check all axis indices are unique and in valid range [0, nd-1]
- Throw informative errors for invalid input
```

### 2. Algorithm Selection

```cpp
if (nd == 0 || nd == 1) {
    // Identity operation - return copy
    return copy
}
else if (nd == 2) {
    if (is_square && inplace) {
        // Use in-place square transpose
        return transpose_2d_square_inplace()
    }
    else {
        // Use optimized 2D transpose
        return transpose_2d_blocked()
    }
}
else {
    // General N-D transpose
    return transpose_nd_general()
}
```

### 3. Performance Optimizations

#### A. Blocked 2D Transpose

For 2D matrices, use a cache-blocked algorithm:

```cpp
const size_t BLOCK_SIZE = 64;  // Tuned for L1 cache

for (size_t ii = 0; ii < rows; ii += BLOCK_SIZE) {
    for (size_t jj = 0; jj < cols; jj += BLOCK_SIZE) {
        // Transpose BLOCK_SIZE x BLOCK_SIZE sub-matrix
        size_t imax = std::min(ii + BLOCK_SIZE, rows);
        size_t jmax = std::min(jj + BLOCK_SIZE, cols);

        for (size_t i = ii; i < imax; i++) {
            for (size_t j = jj; j < jmax; j++) {
                output.f(j, i) = input.f(i, j);
            }
        }
    }
}
```

**Benefits**:
- Improves cache locality for large matrices
- Typically 2-10x faster than naive transpose for > 1000x1000
- Minimal overhead for small matrices

#### B. In-Place Square Transpose

For square matrices, transpose in-place by swapping symmetric elements:

```cpp
for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
        std::swap(data[i * n + j], data[j * n + i]);
    }
}
```

**Benefits**:
- Zero additional memory allocation
- Ideal for large square matrices where memory is constrained

#### C. General N-D Transpose

For arbitrary dimension permutations:

```cpp
// Compute output dimensions and strides
std::vector<size_t> output_dims(nd);
std::vector<size_t> output_strides(nd);
for (size_t i = 0; i < nd; i++) {
    output_dims[i] = input_dims[axes[i]];
}
compute_strides(output_dims, output_strides);

// Copy data with permuted indices
for (size_t i = 0; i < total_elements; i++) {
    // Convert linear index to multi-dimensional indices
    auto indices = linear_to_multidim(i, input_dims);

    // Permute indices
    std::vector<size_t> permuted_indices(nd);
    for (size_t d = 0; d < nd; d++) {
        permuted_indices[d] = indices[axes[d]];
    }

    // Write to output
    output.f(permuted_indices) = input.f(indices);
}
```

**Optimization**: For contiguous sub-arrays, use `memcpy` to copy entire blocks.

### 4. Memory Layout Considerations

The ndarray library uses Fortran-order (column-major) internally:
- First index varies fastest
- 2D array `A(i,j)` is stored as `A[i + j*nrows]`

For transpose operations:
- **Fortran to Fortran**: Natural for transpose (row ↔ column swap)
- **Stride calculation**: Must account for internal storage order

### 5. Error Handling

```cpp
// Invalid axes vector
if (axes.size() != nd()) {
    throw ndarray_error("transpose: axes size must match array dimensions");
}

// Duplicate axes
std::set<size_t> unique_axes(axes.begin(), axes.end());
if (unique_axes.size() != axes.size()) {
    throw ndarray_error("transpose: axes must be unique");
}

// Out of range axes
for (auto axis : axes) {
    if (axis >= nd()) {
        throw ndarray_error("transpose: axis index out of range");
    }
}

// In-place on non-square
if (inplace && (dimf(0) != dimf(1))) {
    throw ndarray_error("transpose_inplace: requires square matrix");
}

// In-place on non-2D
if (inplace && nd() != 2) {
    throw ndarray_error("transpose_inplace: only supported for 2D arrays");
}
```

## Performance Benchmarks (Expected)

### 2D Transpose

| Size | Naive | Blocked | Speedup |
|------|-------|---------|---------|
| 100x100 | 0.02 ms | 0.02 ms | 1.0x |
| 1000x1000 | 3.5 ms | 1.2 ms | 2.9x |
| 4000x4000 | 180 ms | 35 ms | 5.1x |
| 8000x8000 | 850 ms | 150 ms | 5.7x |

### Memory Usage

| Operation | Size | Memory |
|-----------|------|--------|
| Out-of-place | NxM | 2 × N×M×sizeof(T) |
| In-place square | NxN | N×N×sizeof(T) |

## Testing Strategy

### Unit Tests

1. **Basic Functionality**
   - 0D, 1D arrays (identity)
   - 2D transpose (various sizes)
   - 3D, 4D permutations
   - Empty arrays

2. **Correctness**
   - Verify all elements correctly repositioned
   - Test all permutations for small arrays (3D: 3! = 6 permutations)
   - Compare with reference implementations

3. **Edge Cases**
   - Single element arrays
   - Single dimension arrays (e.g., 1×N, N×1)
   - Very large arrays (> 1GB)

4. **Error Handling**
   - Invalid axes (duplicate, out of range, wrong size)
   - In-place on non-square
   - In-place on > 2D

5. **Performance**
   - Benchmark different sizes
   - Verify blocked algorithm is faster for large arrays
   - Memory usage validation

6. **Storage Policies**
   - Test with native_storage
   - Test with xtensor_storage (if available)
   - Test with eigen_storage (if available)

### Test File Structure

```
tests/test_transpose.cpp              # Main test file
tests/test_transpose_performance.cpp  # Performance benchmarks
```

## Future Enhancements

1. **SIMD Optimizations**: Use vectorized instructions for 4-8x speedup
2. **Parallel Transpose**: OpenMP/TBB for multi-threaded transpose
3. **GPU Transpose**: CUDA/HIP kernels for GPU arrays
4. **Strided Views**: Return transpose as strided view without copy (lazy evaluation)
5. **Auto-tuning**: Automatically select block size based on cache size
6. **Conjugate Transpose**: For complex arrays (A^H)

## Implementation Files

- `include/ndarray/transpose.hh` - Header with API declarations
- `include/ndarray/ndarray.hh` - Add member function declarations
- `src/transpose.cpp` - Implementation (if needed for explicit instantiation)
- `tests/test_transpose.cpp` - Comprehensive unit tests
- `tests/test_transpose_performance.cpp` - Performance benchmarks

## References

1. **Cache-Oblivious Algorithms**: Frigo et al., "Cache-Oblivious Algorithms"
2. **Blocked Matrix Transpose**: Goto & van de Geijn, "High-Performance Implementation of BLAS"
3. **NumPy Transpose**: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
4. **Eigen Transpose**: https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html

---

**Document Version**: 1.0
**Date**: 2026-02-25
**Author**: Claude Code Assistant
