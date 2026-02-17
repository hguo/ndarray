# Storage Backend System

The ndarray library supports multiple storage backends through a policy-based design. This allows you to choose the optimal storage implementation for your use case, from simple std::vector to high-performance libraries like xtensor and Eigen.

## Overview

Storage backends are selected at compile-time through template parameters:

```cpp
// Native storage (default) - uses std::vector
ftk::ndarray<double> native_arr;

// xtensor storage - SIMD vectorization and expression templates
ftk::ndarray<double, ftk::xtensor_storage> xt_arr;

// Eigen storage - optimized linear algebra
ftk::ndarray<double, ftk::eigen_storage> eigen_arr;
```

## Available Storage Backends

### 1. Native Storage (Default)

**Features:**
- Uses `std::vector<T>` internally
- No external dependencies
- 100% backward compatible with existing code
- Zero overhead for I/O operations

**When to use:**
- When xtensor/Eigen are not available
- When compatibility is the priority
- For simple data storage and I/O
- When you don't need advanced computational features

**Example:**
```cpp
ftk::ndarray<float> arr;
arr.read_netcdf("data.nc", "temperature");
arr.reshapef(100, 200);
arr.fill(0.0f);
```

### 2. xtensor Storage

**Requirements:**
- xtensor library
- CMake option: `-DNDARRAY_USE_XTENSOR=TRUE`

**Features:**
- SIMD vectorization for element-wise operations
- Lazy evaluation and expression templates
- NumPy-like API compatibility
- Broadcasting support

**When to use:**
- For numerical computations
- When you need broadcasting
- For N-dimensional arrays (any dimension)
- When you want NumPy-like semantics

**Example:**
```cpp
ftk::ndarray_xtensor<double> a, b, c;
a.reshapef(1000, 1000);
b.reshapef(1000, 1000);
c.reshapef(1000, 1000);

// Element-wise operations use SIMD
for (size_t i = 0; i < a.size(); i++) {
  c[i] = a[i] * b[i] + a[i];  // Vectorized
}
```

### 3. Eigen Storage

**Requirements:**
- Eigen3 library
- CMake option: `-DNDARRAY_USE_EIGEN=TRUE`

**Features:**
- Optimized BLAS/LAPACK operations
- Column-major (Fortran) memory layout
- Best for 2D matrices
- Native support for linear algebra operations

**When to use:**
- For linear algebra operations
- For 2D matrices specifically
- When you need eigenvalue/SVD computations
- When interfacing with Fortran code

**Example:**
```cpp
ftk::ndarray_eigen<double> matrix;
matrix.reshapef(500, 500);

// Access underlying Eigen matrix for specialized operations
auto& eigen_mat = matrix.get_matrix();
// Perform Eigen-specific operations
```

## I/O with Different Backends

All I/O operations work seamlessly with all storage backends:

```cpp
// Read with native, write with xtensor
ftk::ndarray<float> native_arr;
native_arr.read_netcdf("input.nc", "field");

ftk::ndarray_xtensor<float> xt_arr = native_arr;  // Convert
// ... fast computation ...

ftk::ndarray<float> result = xt_arr;  // Convert back
result.write_netcdf("output.nc", "result");
```

### I/O Performance Notes

- **Native storage**: Zero-copy I/O (fastest for read/write)
- **xtensor/Eigen storage**: One copy during I/O (small overhead)

For maximum I/O performance with alternative backends, use direct I/O:

```cpp
// Read directly into xtensor storage
ftk::ndarray_xtensor<float> arr;
arr.read_netcdf("data.nc", "field");  // Automatic conversion
```

## Storage Backend Conversion

You can convert between storage backends:

```cpp
// Native to xtensor
ftk::ndarray<double> native_arr(1000);
ftk::ndarray_xtensor<double> xt_arr = native_arr;

// xtensor to Eigen
ftk::ndarray_eigen<double> eigen_arr = xt_arr;

// Back to native
ftk::ndarray<double> native_arr2 = eigen_arr;
```

Conversion involves copying data element-by-element, which has O(n) cost.

## Groups and Streams

Groups and streams are also templated on storage policy:

```cpp
// Native storage group (default)
ftk::ndarray_group<> group;
group.set("temperature", temp_array);

// xtensor storage group
ftk::ndarray_group<ftk::xtensor_storage> xt_group;
xt_group.set("temperature", xt_temp_array);

// Stream with xtensor storage
ftk::stream<ftk::xtensor_storage> s;
s.parse_yaml("config.yaml");
auto g = s.read(0);  // Returns ndarray_group<xtensor_storage>
```

**Important**: All arrays in a group must use the same storage backend.

## Type Aliases

For convenience, use the provided type aliases:

```cpp
// Native storage
ftk::ndarray_native<float>
ftk::ndarray_group_native
ftk::stream_native

// xtensor storage
ftk::ndarray_xtensor<float>
ftk::ndarray_group_xtensor
ftk::stream_xtensor

// Eigen storage
ftk::ndarray_eigen<float>
ftk::ndarray_group_eigen
ftk::stream_eigen
```

## Choosing a Storage Backend

**Use native storage when:**
- Compatibility is the priority
- You only need data storage and I/O
- xtensor/Eigen are not available

**Use xtensor storage when:**
- You want SIMD vectorization
- You're working with N-dimensional arrays
- You want NumPy-like semantics
- You need broadcasting

**Use Eigen storage when:**
- You need linear algebra operations
- You're working primarily with 2D matrices
- You need eigenvalue/SVD computations
- You want BLAS/LAPACK integration

## Migration Guide

Existing code requires **zero changes** - native storage is the default:

```cpp
// This code continues to work exactly as before
ftk::ndarray<double> arr;
arr.read_netcdf("data.nc", "temperature");
```

To use xtensor storage:

```cpp
// Change type declaration
ftk::ndarray<double, ftk::xtensor_storage> arr;
// or use alias
ftk::ndarray_xtensor<double> arr;

// Rest of the code stays the same
arr.read_netcdf("data.nc", "temperature");
```

## Advanced Usage

### Custom Storage Policies

You can implement your own storage policy by providing:

```cpp
struct my_storage {
  template <typename T>
  struct container_type {
    size_t size() const;
    T* data();
    const T* data() const;
    void resize(size_t);
    T& operator[](size_t);
    const T& operator[](size_t) const;
    void fill(T value);

    // Optional: for multi-dimensional reshape support
    void reshape(const std::vector<size_t>&);
  };
};
```

Then use it:

```cpp
ftk::ndarray<double, my_storage> arr;
```

### Accessing Underlying Storage

```cpp
// xtensor: access xt::xarray
ftk::ndarray_xtensor<double> arr;
auto& xarray = arr.storage_.get_xarray();

// Eigen: access Eigen::Matrix
ftk::ndarray_eigen<double> arr;
auto& matrix = arr.storage_.get_matrix();

// Native: access std::vector
ftk::ndarray_native<double> arr;
auto& vec = arr.std_vector();  // Only for native_storage
```

## Limitations

1. **Group homogeneity**: All arrays in a group must use the same storage backend
2. **Conversion overhead**: Converting between backends copies data
3. **I/O overhead**: Non-native backends have one extra copy during I/O
4. **Eigen dimensions**: Eigen storage is optimized for 2D; N-D arrays are flattened

## FAQ

**Q: Can I mix storage backends in the same group?**
A: No, all arrays in a group must use the same storage backend.

**Q: What's the overhead of using xtensor/Eigen storage?**
A: Memory overhead is zero (same layout as native). I/O has one extra copy. Computation is typically faster.

**Q: Can I use my own storage backend?**
A: Yes, implement the storage policy interface (see Advanced Usage).

**Q: Is there a runtime overhead for the policy design?**
A: No, storage backend selection is compile-time (zero runtime overhead).

**Q: Can I use storage backends without xtensor/Eigen installed?**
A: Native storage requires no dependencies. xtensor/Eigen storage only compile when libraries are found.
