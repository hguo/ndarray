# Backend Storage Design

## Overview

ndarray supports **pluggable storage backends** allowing users to choose the underlying data structure. This enables:

1. **Performance optimization** - Use specialized backends for specific use cases
2. **Library interoperability** - Seamless integration with Eigen, xtensor
3. **Memory efficiency** - Leverage optimized storage from mature libraries
4. **Flexibility** - Choose the right tool for the job

## Available Backends

| Backend | Always Available | Library Required | Best For |
|---------|-----------------|------------------|----------|
| `native_backend` | ✅ Yes | None (std::vector) | General purpose, I/O |
| `eigen_backend` | ❌ No | Eigen3 | Linear algebra operations |
| `xtensor_backend` | ❌ No | xtensor | NumPy-like operations |

## Usage

### Native Backend (Default)

The default backend using `std::vector`:

```cpp
#include <ndarray/ndarray.hh>

// Explicit (optional)
ftk::ndarray<double, ftk::native_backend<double>> arr;

// Implicit (default)
ftk::ndarray<double> arr;  // Same as above
```

**Characteristics:**
- Always available
- Good I/O performance
- Standard C++ containers
- Compatible with all ndarray I/O functions

### Eigen Backend

Uses Eigen for storage:

```cpp
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_backend.hh>

// Choose Eigen backend
ftk::ndarray<double, ftk::eigen_backend<double>> arr;
arr.reshapef(1000);

// Access underlying Eigen vector
Eigen::VectorXd& eigen_vec = arr.backend().eigen_vector();

// Use Eigen operations
double norm = eigen_vec.norm();
double mean = eigen_vec.mean();
```

**Characteristics:**
- Optimized for linear algebra
- BLAS/LAPACK acceleration
- Vectorization (SIMD)
- Excellent for matrix operations

**Build requirement:**
```bash
cmake .. -DNDARRAY_USE_EIGEN=ON
```

### xtensor Backend

Uses xtensor for storage:

```cpp
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_backend.hh>

// Choose xtensor backend
ftk::ndarray<float, ftk::xtensor_backend<float>> arr;
arr.reshapef(100, 200);

// Access underlying xtensor array
xt::xarray<float>& xarr = arr.backend().xtensor_array();

// Use xtensor operations
auto sum = xt::sum(xarr);
auto max = xt::amax(xarr);
```

**Characteristics:**
- NumPy-like API
- Broadcasting
- Lazy evaluation
- Universal functions (ufuncs)

**Build requirement:**
```bash
cmake .. -DNDARRAY_USE_XTENSOR=ON
```

## Design Pattern

### Backend Concept

All backends must implement:

```cpp
template <typename T>
struct backend_concept {
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  void resize(size_t n);
  void clear();
  size_t size() const;
  bool empty() const;

  pointer data();
  const_pointer data() const;

  reference operator[](size_t i);
  const_reference operator[](size_t i) const;

  void fill(const T& value);

  auto begin();
  auto end();
  auto begin() const;
  auto end() const;

  std::vector<T> std_vector() const;  // For compatibility
};
```

### Template Signature

The updated ndarray signature:

```cpp
template <typename T, typename Backend = native_backend<T>>
class ndarray {
  Backend backend_;

public:
  // Access backend
  Backend& backend() { return backend_; }
  const Backend& backend() const { return backend_; }

  // Standard interface (delegates to backend)
  T* data() { return backend_.data(); }
  size_t size() const { return backend_.size(); }
  T& operator[](size_t i) { return backend_[i]; }
  // ... etc
};
```

## Practical Examples

### Example 1: I/O with Native Backend

```cpp
// Use native backend for I/O (default behavior)
ftk::ndarray<float> arr;
arr.read_netcdf("data.nc", "temperature");  // Efficient I/O

// Process with native backend
for (size_t i = 0; i < arr.size(); i++) {
  arr[i] *= 2.0f;
}

arr.write_netcdf("output.nc", "result");
```

### Example 2: Linear Algebra with Eigen Backend

```cpp
// Use Eigen backend for linear algebra
ftk::ndarray<double, ftk::eigen_backend<double>> A, b, x;

// Read data
A.read_netcdf("matrix.nc", "A");
b.read_netcdf("vector.nc", "b");

// Solve Ax = b using Eigen
Eigen::Map<Eigen::MatrixXd> A_mat(A.data(), A.dimf(0), A.dimf(1));
Eigen::Map<Eigen::VectorXd> b_vec(b.data(), b.size());

x.reshapef(b.size());
Eigen::Map<Eigen::VectorXd> x_vec(x.data(), x.size());

x_vec = A_mat.colPivHouseholderQr().solve(b_vec);

// Write result
x.write_netcdf("solution.nc", "x");
```

### Example 3: Broadcasting with xtensor Backend

```cpp
// Use xtensor backend for NumPy-like operations
ftk::ndarray<float, ftk::xtensor_backend<float>> arr;
arr.read_netcdf("data.nc", "field");

// Access underlying xtensor
auto& xarr = arr.backend().xtensor_array();

// Reshape for xtensor operations
xarr.reshape({arr.dimf(0), arr.dimf(1), arr.dimf(2)});

// Broadcasting operation
auto mean_per_layer = xt::mean(xarr, {1, 2});  // Average over spatial dims

// Continue with ndarray I/O
arr.write_netcdf("output.nc", "result");
```

### Example 4: Backend Conversion

Convert between backends:

```cpp
// Read with native backend
ftk::ndarray<double> native_arr;
native_arr.read_netcdf("input.nc", "data");

// Convert to Eigen backend
ftk::ndarray<double, ftk::eigen_backend<double>> eigen_arr;
eigen_arr.reshapef(native_arr.dimf(0), native_arr.dimf(1));
std::copy(native_arr.data(), native_arr.data() + native_arr.size(),
          eigen_arr.data());

// Perform Eigen operations
auto& eigen_mat = eigen_arr.backend().eigen_vector();
// ... Eigen operations ...

// Convert back to native for I/O
ftk::ndarray<double> result;
result.reshapef(eigen_arr.dimf(0), eigen_arr.dimf(1));
std::copy(eigen_arr.data(), eigen_arr.data() + eigen_arr.size(),
          result.data());
result.write_netcdf("output.nc", "result");
```

## Performance Considerations

### I/O Performance

- **Native backend**: Best for I/O (direct buffer compatibility)
- **Eigen/xtensor backends**: May require conversion for some I/O operations

### Compute Performance

- **Eigen backend**: Excellent for dense linear algebra (BLAS/LAPACK)
- **xtensor backend**: Good for element-wise and broadcasting operations
- **Native backend**: Standard performance, good for general purpose

### Memory Overhead

All backends have similar memory footprint (contiguous storage).

## Backend Selection Guidelines

### Choose Native Backend When:
- Primary use case is I/O
- Maximum portability required
- No special operations needed
- **Default choice for most scientific I/O workflows**

### Choose Eigen Backend When:
- Heavy linear algebra (matrix multiplication, decomposition, solvers)
- Need BLAS/LAPACK acceleration
- Working with 1D or 2D data
- Integration with existing Eigen code

### Choose xtensor Backend When:
- Need NumPy-like operations
- Broadcasting is important
- Working with high-dimensional data
- Integration with existing xtensor code

## Implementation Status

### Current (v0.0.1)

- ✅ Native backend fully implemented
- ✅ Backend infrastructure designed
- ❌ Eigen backend implemented (storage policy only)
- ❌ xtensor backend implemented (storage policy only)
- ❌ Template parameter plumbing in ndarray class (requires refactoring)

### Roadmap

**Phase 1** (Current):
- Backend policy classes (done)
- Documentation (this file)

**Phase 2** (Next):
- Refactor `ndarray` to accept Backend template parameter
- Update all I/O methods to work with backend concept
- Add backend conversion utilities

**Phase 3**:
- Optimize I/O for each backend
- Add backend-specific optimizations
- Performance benchmarks

## Migration Guide

### From Current API (Pre-Backend)

```cpp
// Old API (current)
ftk::ndarray<double> arr;
arr.reshapef(100, 200);
std::vector<double>& vec = arr.p;  // Direct access to std::vector

// New API (with backends)
ftk::ndarray<double, ftk::native_backend<double>> arr;
arr.reshapef(100, 200);
std::vector<double>& vec = arr.backend().std_vector();  // Access through backend
```

**Backward compatibility:**
- Default template parameter ensures existing code works
- `.p` member can be aliased to `backend().std_vector()`

## Custom Backend

Advanced users can implement custom backends:

```cpp
template <typename T>
struct my_custom_backend {
  // ... implement backend_concept interface ...
};

// Use custom backend
ftk::ndarray<float, my_custom_backend<float>> arr;
```

**Requirements:**
- Implement all backend_concept methods
- Provide contiguous memory layout
- Support iterators

## Best Practices

1. **Default to native backend** for I/O-heavy workflows
2. **Convert to specialized backend** for compute-intensive sections
3. **Profile before optimizing** - measure actual performance
4. **Consider memory copies** - backend conversion has O(n) cost
5. **Document backend choice** - make it explicit in code comments

## See Also

- [BACKENDS.md](BACKENDS.md) - Eigen/xtensor interoperability (conversion-based)
- [ZERO_COPY_OPTIMIZATION.md](ZERO_COPY_OPTIMIZATION.md) - Memory optimization
- [GPU_SUPPORT.md](GPU_SUPPORT.md) - Device memory management
