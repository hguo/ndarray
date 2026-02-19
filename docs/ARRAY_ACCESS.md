# Array Access Functions Guide

## Overview

This document explains how to use ndarray's various access functions for querying dimensions and accessing elements. Understanding these functions is critical for correct usage of the library.

---

## Quick Reference

| Function | Purpose | Example |
|----------|---------|---------|
| `nd()` | Number of dimensions | `arr.nd() == 3` |
| `size()` | Total number of elements | `arr.size() == 1000` |
| `dimf(i)` | Get dimension (Fortran order) | `arr.dimf(0)` returns first dim |
| `dimc(i)` | Get dimension (C order - reversed) | `arr.dimc(0)` returns last dim |
| `shapef()` | All dimensions (Fortran order) | `[nx, ny, nz]` |
| `shapec()` | All dimensions (C order - reversed) | `[nz, ny, nx]` |
| `f(i,j,k)` | Element access (Fortran order) | First index varies fastest |
| `c(k,j,i)` | Element access (C order) | Last index varies fastest |
| `at(...)` | Element access (vector indices) | `arr.at({i,j,k})` |
| `operator[]` | Linear access | `arr[idx]` - row-major |

---

## 1. Dimension Query Functions

### 1.1 `nd()` - Number of Dimensions

```cpp
size_t nd() const;
```

**Returns**: Total number of dimensions in the array.

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);

std::cout << arr.nd();  // Output: 3
```

**Use cases**:
- Check dimensionality before operations
- Generic code that adapts to dimension count
- Validation

---

### 1.2 `size()` - Total Number of Elements

```cpp
size_t size() const;
```

**Returns**: Product of all dimensions (total array elements).

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);

std::cout << arr.size();  // Output: 6000 (10 * 20 * 30)
```

**Use cases**:
- Allocate buffers
- Loop over all elements
- Memory calculations

**Important**: `size()` counts ALL elements including component dimensions:
```cpp
// Vector field: 3 components per spatial point
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200);  // [3 components, 100x200 spatial]
velocity.set_multicomponents(1);

std::cout << velocity.size();  // 60000 (3 * 100 * 200)
```

---

### 1.3 `dimf(i)` - Fortran-Order Dimension Access

```cpp
size_t dimf(size_t i) const;
```

**Returns**: Dimension at index `i` in **storage order** (Fortran/column-major).

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);

std::cout << arr.dimf(0);  // 10 (first dimension)
std::cout << arr.dimf(1);  // 20 (second dimension)
std::cout << arr.dimf(2);  // 30 (third dimension)
```

**Memory layout**:
```
Memory: [all i for j=0,k=0 | all i for j=1,k=0 | ... ]
         ^^^^^^^^^^^^^^^^^
         First index (i) varies fastest
```

**Use when**:
- Working with Fortran-order data
- Reading from NetCDF/HDF5 (typically Fortran-order)
- Column-major algorithms

---

### 1.4 `dimc(i)` - C-Order Dimension Access

```cpp
size_t dimc(size_t i) const;
```

**Returns**: Dimension at index `i` in **C declaration order** (reversed).

**Implementation**:
```cpp
size_t dimc(size_t i) const { return dims[dims.size() - i - 1]; }
```

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);  // Internal: [10, 20, 30]

std::cout << arr.dimc(0);  // 30 (LAST dimension)
std::cout << arr.dimc(1);  // 20 (middle dimension)
std::cout << arr.dimc(2);  // 10 (FIRST dimension)
```

**Why reversed?** Matches C array declaration order:
```cpp
// C declaration: float arr[30][20][10]
// Corresponds to: dimc(0)=30, dimc(1)=20, dimc(2)=10
```

**Use when**:
- Interfacing with C/C++ row-major code
- NumPy C-order arrays
- Row-major algorithms

---

### 1.5 `shapef()` - Full Shape (Fortran Order)

```cpp
const std::vector<size_t>& shapef() const;
size_t shapef(size_t i) const;  // Equivalent to dimf(i)
```

**Returns**: Vector of all dimensions in storage order.

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);

auto shape = arr.shapef();
// shape = {10, 20, 30}

for (size_t d : shape) {
  std::cout << d << " ";  // Output: 10 20 30
}
```

**Use cases**:
- Pass shape to other functions
- Iterate over dimensions
- Shape validation

---

### 1.6 `shapec()` - Full Shape (C Order - Reversed)

```cpp
std::vector<size_t> shapec() const;
size_t shapec(size_t i) const;  // Equivalent to dimc(i)
```

**Returns**: Vector of dimensions in **reversed** order.

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);

auto shape_c = arr.shapec();
// shape_c = {30, 20, 10}  (reversed!)

for (size_t d : shape_c) {
  std::cout << d << " ";  // Output: 30 20 10
}
```

**Matching C declarations**:
```cpp
// ndarray internal: [10, 20, 30]
// shapec() returns: {30, 20, 10}
// Matches C: float arr[30][20][10]
```

---

## 2. Element Access Functions

### 2.1 `f(...)` - Fortran-Order Element Access

```cpp
T& f(size_t i0);
T& f(size_t i0, size_t i1);
T& f(size_t i0, size_t i1, size_t i2);
// ... up to 10 dimensions
```

**Access pattern**: **First index varies fastest** (column-major).

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(3, 4, 5);  // [nx=3, ny=4, nz=5]

// Fill array
for (size_t k = 0; k < 5; k++)
  for (size_t j = 0; j < 4; j++)
    for (size_t i = 0; i < 3; i++)
      arr.f(i, j, k) = i + j*10 + k*100;

// Access element at (1, 2, 3)
float val = arr.f(1, 2, 3);  // = 1 + 20 + 300 = 321
```

**Memory layout**:
```
f(0,0,0), f(1,0,0), f(2,0,0),  <- First index changes
f(0,1,0), f(1,1,0), f(2,1,0),
f(0,2,0), ...
```

**Index bounds**:
```cpp
arr.reshapef(nx, ny, nz);

// Valid indices:
// i ∈ [0, dimf(0)) = [0, nx)
// j ∈ [0, dimf(1)) = [0, ny)
// k ∈ [0, dimf(2)) = [0, nz)

arr.f(i, j, k);  // i < nx, j < ny, k < nz
```

**Use when**:
- Working with Fortran-order data
- NetCDF/HDF5 I/O (typically Fortran-order)
- Scientific computing (common convention)

---

### 2.2 `c(...)` - C-Order Element Access

```cpp
T& c(size_t i0);
T& c(size_t i0, size_t i1);
T& c(size_t i0, size_t i1, size_t i2);
// ... up to 10 dimensions
```

**Access pattern**: **Last index varies fastest** (row-major).

**IMPORTANT**: `c()` is **consistent with NumPy's default C-order behavior**.

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(3, 4, 5);  // Internal: [3, 4, 5]

// C-style access: indices REVERSED
// c(k, j, i) accesses same element as f(i, j, k)

arr.c(0, 0, 0) == arr.f(0, 0, 0);  // true
arr.c(3, 2, 1) == arr.f(1, 2, 3);  // true
```

**Index bounds**:
```cpp
arr.reshapef(nx, ny, nz);  // [3, 4, 5]

// C-style access: c(k, j, i)
// k ∈ [0, dimc(0)) = [0, nz) = [0, 5)
// j ∈ [0, dimc(1)) = [0, ny) = [0, 4)
// i ∈ [0, dimc(2)) = [0, nx) = [0, 3)

arr.c(k, j, i);  // REVERSED indices!
```

**Matching C arrays**:
```cpp
// C declaration: float c_arr[5][4][3];
// ndarray: arr.reshapef(3, 4, 5);

// Access patterns match:
c_arr[2][1][0] == arr.c(2, 1, 0);
```

**Use when**:
- Interfacing with C/C++ code
- NumPy C-order arrays
- Row-major algorithms

**WARNING**: Index order is **reversed** compared to `f()`!

---

### 2.3 `at(...)` - Vector Index Access (DEPRECATED)

```cpp
[[deprecated]] T& at(const std::vector<size_t>& idx);
```

**Access with vector of indices** (Fortran-order).

**DEPRECATED**: Use `f()` instead for Fortran-order or `c()` for NumPy/C-order compatibility.

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);

std::vector<size_t> idx = {5, 10, 15};
float val = arr.at(idx);  // Same as arr.f(5, 10, 15) - DEPRECATED!

// Use f() instead:
float val = arr.f(5, 10, 15);  // Fortran-order (first index varies fastest)

// Or use c() for NumPy compatibility:
float val = arr.c(15, 10, 5);  // C-order (last index varies fastest)
```

**Note**: `at()` uses Fortran-order (same as `f()`), NOT C-order like NumPy's default.

---

### 2.4 `operator[]` - Linear Indexing

```cpp
T& operator[](size_t idx);
const T& operator[](size_t idx) const;
```

**Direct access to underlying storage** (1D linear index).

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(3, 4);  // 12 elements total

// Linear access
for (size_t i = 0; i < arr.size(); i++) {
  arr[i] = i * 1.5f;
}

// arr[0], arr[1], arr[2], ... arr[11]
```

**Relationship to multi-dimensional indices**:
```cpp
// For reshapef(nx, ny, nz):
// Linear index: idx = i + j*nx + k*nx*ny

arr.f(i, j, k) == arr[i + j*arr.dimf(0) + k*arr.dimf(0)*arr.dimf(1)];
```

**Use when**:
- Iterating over all elements
- Interfacing with raw pointers
- Performance-critical loops

**Warning**: Index calculation is manual and error-prone for multi-dimensional access.

---

## 3. Practical Examples

### Example 1: Reading NetCDF and Accessing Data

```cpp
#include <ndarray/ndarray.hh>

int main() {
  // Read 3D temperature field from NetCDF
  ftk::ndarray<float> temp;
  temp.read_netcdf("climate.nc", "temperature");

  // Query dimensions
  std::cout << "Dimensions: " << temp.nd() << "D array" << std::endl;
  std::cout << "Shape: ";
  for (size_t i = 0; i < temp.nd(); i++) {
    std::cout << temp.dimf(i);
    if (i < temp.nd()-1) std::cout << " x ";
  }
  std::cout << std::endl;

  // Example: [lon, lat, time] = [360, 180, 120]
  size_t nlon = temp.dimf(0);  // 360
  size_t nlat = temp.dimf(1);  // 180
  size_t ntime = temp.dimf(2); // 120

  // Access specific point using Fortran-order
  float temp_nyc_jan = temp.f(285, 50, 0);  // NYC, January, timestep 0

  // Compute zonal mean (average over longitude)
  for (size_t t = 0; t < ntime; t++) {
    for (size_t lat = 0; lat < nlat; lat++) {
      float sum = 0.0f;
      for (size_t lon = 0; lon < nlon; lon++) {
        sum += temp.f(lon, lat, t);
      }
      float mean = sum / nlon;
      std::cout << "Timestep " << t << ", Lat " << lat << ": " << mean << std::endl;
    }
  }

  return 0;
}
```

### Example 2: C-Style Array Interop

```cpp
// C function expecting row-major 3D array
extern "C" void process_3d_array(float* data, int nz, int ny, int nx);

int main() {
  // Create ndarray with Fortran-order
  ftk::ndarray<float> arr;
  arr.reshapef(100, 200, 50);  // [nx=100, ny=200, nz=50]

  // Fill using Fortran-order
  for (size_t k = 0; k < 50; k++)
    for (size_t j = 0; j < 200; j++)
      for (size_t i = 0; i < 100; i++)
        arr.f(i, j, k) = i + j + k;

  // Access dimensions in C-order for C function
  int nz = arr.dimc(0);  // 50
  int ny = arr.dimc(1);  // 200
  int nx = arr.dimc(2);  // 100

  // Pass to C function (note: still column-major storage!)
  process_3d_array(arr.data(), nz, ny, nx);

  return 0;
}
```

### Example 3: Generic Dimension Code

```cpp
template <typename T>
void print_array_info(const ftk::ndarray<T>& arr) {
  std::cout << "Array info:" << std::endl;
  std::cout << "  Dimensions: " << arr.nd() << "D" << std::endl;
  std::cout << "  Total elements: " << arr.size() << std::endl;

  std::cout << "  Shape (Fortran-order): ";
  auto sf = arr.shapef();
  for (size_t i = 0; i < sf.size(); i++) {
    std::cout << sf[i];
    if (i < sf.size()-1) std::cout << " x ";
  }
  std::cout << std::endl;

  std::cout << "  Shape (C-order): ";
  auto sc = arr.shapec();
  for (size_t i = 0; i < sc.size(); i++) {
    std::cout << sc[i];
    if (i < sc.size()-1) std::cout << " x ";
  }
  std::cout << std::endl;
}

int main() {
  ftk::ndarray<double> arr;
  arr.reshapef(10, 20, 30, 40);

  print_array_info(arr);

  // Output:
  // Array info:
  //   Dimensions: 4D
  //   Total elements: 240000
  //   Shape (Fortran-order): 10 x 20 x 30 x 40
  //   Shape (C-order): 40 x 30 x 20 x 10

  return 0;
}
```

### Example 4: Multi-Component Arrays

```cpp
// 3D velocity field: 3 components (vx, vy, vz) per point
ftk::ndarray<float> velocity;
velocity.reshapef(3, 64, 64, 64);  // [3 components, 64x64x64 spatial]
velocity.set_multicomponents(1);   // First dimension is components

// Query dimensions
std::cout << "Total dimensions: " << velocity.nd() << std::endl;  // 4
std::cout << "Component dimensions: " << velocity.multicomponents() << std::endl;  // 1
std::cout << "Spatial dimensions: " << velocity.nd() - velocity.multicomponents() << std::endl;  // 3

// Access velocity at spatial location (32, 32, 32)
float vx = velocity.f(0, 32, 32, 32);  // x-component
float vy = velocity.f(1, 32, 32, 32);  // y-component
float vz = velocity.f(2, 32, 32, 32);  // z-component

// Compute velocity magnitude at each point
for (size_t k = 0; k < 64; k++) {
  for (size_t j = 0; j < 64; j++) {
    for (size_t i = 0; i < 64; i++) {
      float vx = velocity.f(0, i, j, k);
      float vy = velocity.f(1, i, j, k);
      float vz = velocity.f(2, i, j, k);
      float mag = std::sqrt(vx*vx + vy*vy + vz*vz);
      std::cout << "Velocity magnitude at (" << i << "," << j << "," << k << "): " << mag << std::endl;
    }
  }
}
```

---

## 4. Common Pitfalls

### Pitfall 1: Mixing `f()` and `dimc()`

```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);

// WRONG: Using C-order dimensions with Fortran-order access
for (size_t k = 0; k < arr.dimc(0); k++)  // dimc(0) = 30
  for (size_t j = 0; j < arr.dimc(1); j++)  // dimc(1) = 20
    for (size_t i = 0; i < arr.dimc(2); i++)  // dimc(2) = 10
      arr.f(i, j, k) = ...;  // WRONG BOUNDS!

// CORRECT: Use consistent convention
for (size_t k = 0; k < arr.dimf(2); k++)  // dimf(2) = 30
  for (size_t j = 0; j < arr.dimf(1); j++)  // dimf(1) = 20
    for (size_t i = 0; i < arr.dimf(0); i++)  // dimf(0) = 10
      arr.f(i, j, k) = ...;
```

### Pitfall 2: Index Order Confusion with `c()`

```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20, 30);

// WRONG: Using f() index order with c()
float val = arr.c(5, 10, 15);  // Accesses arr.f(15, 10, 5) NOT arr.f(5, 10, 15)!

// CORRECT: Reverse indices for c()
float val = arr.c(15, 10, 5);  // Accesses arr.f(5, 10, 15)
```

### Pitfall 3: Out-of-Bounds Access

```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20);

// WRONG: Indices out of bounds
arr.f(10, 0);  // UNDEFINED! dimf(0) = 10, valid range [0, 10)
arr.f(0, 20);  // UNDEFINED! dimf(1) = 20, valid range [0, 20)

// CORRECT: Check bounds
if (i < arr.dimf(0) && j < arr.dimf(1)) {
  arr.f(i, j) = ...;
}
```

### Pitfall 4: Forgetting Component Dimensions

```cpp
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 100);
velocity.set_multicomponents(1);

// WRONG: Treating as 3D spatial
for (size_t k = 0; k < velocity.dimf(2); k++)  // Only 2 spatial dims!
  for (size_t j = 0; j < velocity.dimf(1); j++)
    for (size_t i = 0; i < velocity.dimf(0); i++)
      // ...

// CORRECT: Account for component dimension
size_t ncomp = velocity.dimf(0);  // 3 components
size_t nx = velocity.dimf(1);     // 100
size_t ny = velocity.dimf(2);     // 100

for (size_t c = 0; c < ncomp; c++)
  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++)
      velocity.f(c, i, j) = ...;
```

---

## 5. Performance Considerations

### Cache Efficiency

**Fortran-order (`f()`)**: First index varies fastest
```cpp
// GOOD: Sequential memory access
for (size_t k = 0; k < nz; k++)
  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++)  // Inner loop: sequential
      sum += arr.f(i, j, k);

// BAD: Strided memory access
for (size_t i = 0; i < nx; i++)
  for (size_t j = 0; j < ny; j++)
    for (size_t k = 0; k < nz; k++)  // Inner loop: large stride
      sum += arr.f(i, j, k);
```

**Linear access** (`operator[]`): Always sequential
```cpp
// BEST: Maximum cache efficiency
for (size_t idx = 0; idx < arr.size(); idx++) {
  sum += arr[idx];
}
```

---

## 6. Summary Table

| Function | Order | Returns | Use Case |
|----------|-------|---------|----------|
| `nd()` | N/A | Dimension count | Query array rank |
| `size()` | N/A | Total elements | Memory allocation |
| `dimf(i)` | Fortran | Single dimension | Column-major code |
| `dimc(i)` | C | Single dimension (reversed) | Row-major code |
| `shapef()` | Fortran | All dimensions | Full shape info |
| `shapec()` | C | All dimensions (reversed) | C-style shape |
| `f(...)` | Fortran | Element reference | NetCDF/Fortran I/O |
| `c(...)` | C | Element reference | **NumPy/C I/O** |
| `at(vec)` | Fortran | Element reference (DEPRECATED) | Use f() or c() instead |
| `operator[]` | Linear | Element reference | Fast iteration |

---

## 7. Related Documentation

- [FORTRAN_C_CONVENTIONS.md](./FORTRAN_C_CONVENTIONS.md) - Detailed F/C ordering explanation
- [MULTICOMPONENT_ARRAYS.md](./MULTICOMPONENT_ARRAYS.md) - Component dimension handling
- [TIME_DIMENSION.md](./TIME_DIMENSION.md) - Time dimension specifics
- [ARRAY_INDEXING.md](./ARRAY_INDEXING.md) - Indexing patterns and examples

---

**Last Updated**: 2026-02-12
