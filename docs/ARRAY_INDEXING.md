# Array Indexing: F-order vs C-order

## Overview

The ndarray library provides two indexing schemes for accessing multi-dimensional arrays:
- **f()**: Column-major (Fortran-style) indexing
- **c()**: Row-major (C-style) indexing

Both methods access the same underlying contiguous memory but use different index calculations.

## Memory Layout

All ndarrays are stored in **contiguous memory** using a 1D std::vector<T>. The `reshapef()` function computes strides for column-major indexing:

```cpp
void reshapef(const std::vector<size_t>& dims) {
  s[0] = 1;
  s[1] = dims[0];
  s[2] = dims[0] * dims[1];
  s[3] = dims[0] * dims[1] * dims[2];
  // ... and so on
}
```

## Column-Major Indexing: f()

**Definition**: The first index varies fastest in memory.

For a 2D array with dimensions `(n0, n1)`:
```
f(i0, i1) = memory[i0 + i1*n0]
```

For a 3D array with dimensions `(n0, n1, n2)`:
```
f(i0, i1, i2) = memory[i0 + i1*n0 + i2*n0*n1]
```

**Memory Layout Example** (2D array 3x4):
```
f(0,0) f(1,0) f(2,0) | f(0,1) f(1,1) f(2,1) | f(0,2) f(1,2) f(2,2) | f(0,3) f(1,3) f(2,3)
```

This matches **Fortran** and **MATLAB** column-major conventions where columns are contiguous in memory.

## Row-Major Indexing: c()

**Definition**: The last index varies fastest in memory.

For a 2D array with dimensions `(n0, n1)`:
```
c(i0, i1) = memory[i1 + i0*n1]
```

For a 3D array with dimensions `(n0, n1, n2)`:
```
c(i0, i1, i2) = memory[i2 + i1*n2 + i0*n1*n2]
```

**Memory Layout Example** (2D array 3x4):
```
c(0,0) c(0,1) c(0,2) c(0,3) | c(1,0) c(1,1) c(1,2) c(1,3) | c(2,0) c(2,1) c(2,2) c(2,3)
```

This matches **C**, **C++**, and **NumPy** (default) row-major conventions where rows are contiguous in memory.

## When to Use Each

### Use f() when:
- Interoperating with Fortran code
- Reading NetCDF/HDF5 data written by Fortran programs
- Working with MPAS-Ocean or other Fortran-based climate models
- You want the first index to iterate over contiguous memory (cache-friendly)

### Use c() when:
- Interoperating with C/C++ code
- Reading data in standard C row-major format
- Working with NumPy arrays (default order)
- You want the last index to iterate over contiguous memory (cache-friendly)

## Performance Considerations

**Cache Performance**: Accessing contiguous memory is much faster due to cache locality.

For column-major storage (f):
```cpp
// GOOD: i0 varies, accesses contiguous memory
for (size_t i1 = 0; i1 < n1; i1++)
  for (size_t i0 = 0; i0 < n0; i0++)
    process(arr.f(i0, i1));

// BAD: i1 varies in inner loop, jumps n0 elements each time
for (size_t i0 = 0; i0 < n0; i0++)
  for (size_t i1 = 0; i1 < n1; i1++)
    process(arr.f(i0, i1));
```

For row-major access (c):
```cpp
// GOOD: i1 varies, accesses contiguous memory
for (size_t i0 = 0; i0 < n0; i0++)
  for (size_t i1 = 0; i1 < n1; i1++)
    process(arr.c(i0, i1));

// BAD: i0 varies in inner loop, jumps n1 elements each time
for (size_t i1 = 0; i1 < n1; i1++)
  for (size_t i0 = 0; i0 < n0; i0++)
    process(arr.c(i0, i1));
```

## Examples

### 2D Array Example

```cpp
ndarray<double> arr;
arr.reshapef(3, 4);  // 3 rows, 4 columns

// Initialize with column-major access
for (size_t j = 0; j < 4; j++)
  for (size_t i = 0; i < 3; i++)
    arr.f(i, j) = i + j * 10;

// Result in memory:
// [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32]

// Access the same data with row-major
for (size_t i = 0; i < 3; i++)
  for (size_t j = 0; j < 4; j++)
    std::cout << arr.c(i, j) << " ";  // Same values, different indexing

// f(1, 2) and c(1, 2) access DIFFERENT memory locations!
// f(1, 2) = memory[1 + 2*3] = memory[7] = 21
// c(1, 2) = memory[2 + 1*4] = memory[6] = 20
```

### 3D Array Example

```cpp
ndarray<int> arr3d;
arr3d.reshapef(2, 3, 4);  // dimensions: 2 x 3 x 4

// Column-major: first index varies fastest
arr3d.f(0, 0, 0);  // memory[0]
arr3d.f(1, 0, 0);  // memory[1]
arr3d.f(0, 1, 0);  // memory[2]
arr3d.f(1, 1, 0);  // memory[3]
arr3d.f(0, 2, 0);  // memory[4]
arr3d.f(1, 2, 0);  // memory[5]
arr3d.f(0, 0, 1);  // memory[6]

// Row-major: last index varies fastest
arr3d.c(0, 0, 0);  // memory[0]
arr3d.c(0, 0, 1);  // memory[1]
arr3d.c(0, 0, 2);  // memory[2]
arr3d.c(0, 0, 3);  // memory[3]
arr3d.c(0, 1, 0);  // memory[4]
arr3d.c(0, 1, 1);  // memory[5]
arr3d.c(0, 1, 2);  // memory[6]
```

## Relationship to reshape()

The library provides `reshapef()` (Fortran-style reshape) which computes strides for column-major indexing. When you use:

```cpp
arr.reshapef(n0, n1, n2);
```

The strides are: `s = [1, n0, n0*n1]`, which makes f() access contiguous when the first index varies.

## Interoperability

### NetCDF Files
NetCDF typically stores multi-dimensional data in column-major order (Fortran convention). Use f() for natural indexing:

```cpp
ndarray<float> temperature;
temperature.from_netcdf("ocean.nc", "temperature");
// temperature.f(x, y, z, t) matches NetCDF dimension order
```

### NumPy Arrays
NumPy defaults to row-major (C order). When interfacing with Python:

```cpp
// Python: arr = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
// C++: use c() to match NumPy's indexing
arr.c(0, 1);  // matches arr[0, 1] in NumPy
```

## Summary

| Feature | f() - Column-major | c() - Row-major |
|---------|-------------------|-----------------|
| First index varies | Fastest | Slowest |
| Last index varies | Slowest | Fastest |
| Memory order | Fortran-like | C-like |
| Contiguous access | Vary first index | Vary last index |
| Common in | Fortran, MATLAB, NetCDF | C, C++, NumPy |
| Formula (2D) | `i0 + i1*n0` | `i1 + i0*n1` |

Both methods access the same underlying data - the choice depends on your use case and the origin of your data.
