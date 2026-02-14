# Fortran vs C Conventions in ndarray

This document clarifies the differences between Fortran-style (`f` suffix) and C-style (`c` suffix) functions in ndarray.

## Core Concept

ndarray provides **dual interfaces** for dimension specification and access:
- **Fortran-style** (`f` suffix): Column-major convention
- **C-style** (`c` suffix): Row-major convention with **reversed dimension order**

## Dimension Ordering

### dimf() - Fortran-style Dimension Access

Returns dimensions in **storage order** (as stored in memory):

```cpp
size_t dimf(size_t i) const { return dims[i]; }
```

**Example:**
```cpp
ndarray<double> arr;
arr.reshapef(3, 4, 5);  // Shape: [3, 4, 5]

arr.dimf(0);  // 3
arr.dimf(1);  // 4
arr.dimf(2);  // 5
```

### dimc() - C-style Dimension Access

Returns dimensions in **reverse order**:

```cpp
size_t dimc(size_t i) const { return dims[dims.size() - i - 1]; }
```

**Example:**
```cpp
ndarray<double> arr;
arr.reshapef(3, 4, 5);  // Internal storage: [3, 4, 5]

arr.dimc(0);  // 5  (last dimension)
arr.dimc(1);  // 4  (middle dimension)
arr.dimc(2);  // 3  (first dimension)
```

## Why the Reversal?

The reversal reflects the different conventions:

| Convention | Natural Order | Fast-Varying Index |
|------------|---------------|-------------------|
| Fortran | `(nx, ny, nz)` | First (nx) |
| C | `[nz][ny][nx]` | Last (nx) |

**Key insight**: `dimc()` presents dimensions in the order a C programmer would declare them in multi-dimensional array syntax.

## Shape Access

### shapef() - Fortran-style Shape

```cpp
const std::vector<size_t>& shapef() const { return dims; }
size_t shapef(size_t i) const { return dimf(i); }
```

Returns dimensions as stored.

### shapec() - C-style Shape

```cpp
const std::vector<size_t> shapec() const {
  std::vector<size_t> dc(dims);
  std::reverse(dc.begin(), dc.end());
  return dc;
}
size_t shapec(size_t i) const { return dimc(i); }
```

Returns **reversed** dimensions.

**Example:**
```cpp
arr.reshapef(3, 4, 5);

auto sf = arr.shapef();  // [3, 4, 5]
auto sc = arr.shapec();  // [5, 4, 3]
```

## Reshaping

### reshapef() - Fortran-style Reshape

Reshapes array with dimensions in **storage order**:

```cpp
virtual void reshapef(const std::vector<size_t>& dims_) = 0;
```

This is the **primary** reshape function. All data is actually stored using this convention.

**Example:**
```cpp
arr.reshapef(10, 20, 30);
// Internal dims: [10, 20, 30]
// Memory layout: 10 varies fastest (column-major)
```

### reshapec() - C-style Reshape

Reshapes array with dimensions in **C declaration order**:

```cpp
void reshapec(const std::vector<size_t>& dims_) {
  std::vector<size_t> dims(dims_);
  std::reverse(dims.begin(), dims.end());  // Reverse first!
  reshapef(dims);  // Then call reshapef
}
```

**Key**: `reshapec()` **reverses** dimensions before calling `reshapef()`.

**Example:**
```cpp
arr.reshapec(30, 20, 10);
// Reversed to: [10, 20, 30]
// Internal dims: [10, 20, 30]
// Same result as reshapef(10, 20, 30)!
```

## Complete Example

```cpp
ndarray<double> arr;

// Fortran-style: dimensions in storage order
arr.reshapef(3, 4, 5);

std::cout << "nd() = " << arr.nd() << std::endl;  // 3

// Fortran-style access
std::cout << "dimf(0) = " << arr.dimf(0) << std::endl;  // 3
std::cout << "dimf(1) = " << arr.dimf(1) << std::endl;  // 4
std::cout << "dimf(2) = " << arr.dimf(2) << std::endl;  // 5

// C-style access (reversed)
std::cout << "dimc(0) = " << arr.dimc(0) << std::endl;  // 5
std::cout << "dimc(1) = " << arr.dimc(1) << std::endl;  // 4
std::cout << "dimc(2) = " << arr.dimc(2) << std::endl;  // 3

// Shape vectors
auto sf = arr.shapef();  // [3, 4, 5]
auto sc = arr.shapec();  // [5, 4, 3]
```

## Relationship to f() and c() Indexing

The `dimf`/`dimc` naming matches the `f()`/`c()` element access functions:

```cpp
// For array shaped as reshapef(3, 4, 5):

// Fortran-style: first index varies fastest
double val = arr.f(i, j, k);  // i ∈ [0, dimf(0)), j ∈ [0, dimf(1)), k ∈ [0, dimf(2))
                              // i ∈ [0, 3), j ∈ [0, 4), k ∈ [0, 5)

// C-style: last index varies fastest
double val = arr.c(k, j, i);  // k ∈ [0, dimc(0)), j ∈ [0, dimc(1)), i ∈ [0, dimc(2))
                              // k ∈ [0, 5), j ∈ [0, 4), i ∈ [0, 3)
```

**Note**: The index order is **reversed** for `c()` to match C's row-major convention.

## Use Cases

### When to Use Fortran-Style (f)

1. **Scientific Computing**: Fortran conventions dominate HPC
2. **NetCDF/HDF5**: These formats use column-major (Fortran) order
3. **MATLAB Interop**: MATLAB uses column-major
4. **Internal Consistency**: All ndarray internals use Fortran order

**Recommendation**: Use `reshapef()` and `dimf()` by default.

### When to Use C-Style (c)

1. **C/C++ Interop**: Matching C array declarations
2. **NumPy (C-order) Interop**: When working with C-ordered NumPy arrays
3. **User Familiarity**: C programmers expect `[nz][ny][nx]` order

**Note**: C-style functions are convenience wrappers. Internally, everything is still Fortran-order.

## Common Confusions

### Confusion 1: reshapec() vs reshapef() with reversed args

```cpp
// These are THE SAME:
arr.reshapef(3, 4, 5);
arr.reshapec(5, 4, 3);  // Reversed!

// Both result in internal dims: [3, 4, 5]
```

### Confusion 2: dimc() indices

```cpp
arr.reshapef(10, 20, 30);

// WRONG interpretation:
// dimc(0) = 10?  NO!

// CORRECT:
// dimc(0) = 30  (last Fortran dimension)
// dimc(1) = 20  (middle dimension)
// dimc(2) = 10  (first Fortran dimension)
```

### Confusion 3: Thinking c() changes memory layout

```cpp
// Memory layout is ALWAYS column-major (Fortran)
// c() and reshapec() just provide different *views*
// of the same underlying storage
```

## Practical Examples

### Example 1: 3D Array from C Perspective

A C programmer wants a `[5][4][3]` array:

```cpp
// C declaration: double arr[5][4][3]
// Dimensions: nz=5, ny=4, nx=3
// Last index (nx) varies fastest in C

// Option 1: Use reshapec()
arr.reshapec(5, 4, 3);  // C-style: [nz, ny, nx]

// Option 2: Use reshapef() with reversed order
arr.reshapef(3, 4, 5);  // Fortran-style: [nx, ny, nz]

// Both are equivalent! Internal storage: [3, 4, 5]

// Access with c():
double val = arr.c(z, y, x);  // C-style indexing

// Or access with f():
double val = arr.f(x, y, z);  // Fortran-style indexing
```

### Example 2: Reading NetCDF Data

NetCDF uses Fortran (column-major) order:

```cpp
// NetCDF dimensions: (time, z, y, x) = (100, 50, 200, 300)
// First dimension (time) is unlimited, often

arr.read_netcdf(ncid, "temperature");

// NetCDF's (time, z, y, x) becomes internal [time, z, y, x]
// Naturally maps to Fortran order
std::cout << "Time steps: " << arr.dimf(0) << std::endl;  // 100
std::cout << "NZ: " << arr.dimf(1) << std::endl;  // 50
std::cout << "NY: " << arr.dimf(2) << std::endl;  // 200
std::cout << "NX: " << arr.dimf(3) << std::endl;  // 300

// Access:
double T = arr.f(t, z, y, x);
```

### Example 3: Interfacing with NumPy

```python
# Python (NumPy, C-order):
import numpy as np
arr_np = np.zeros((50, 40, 30), order='C')  # C-order: last index varies fastest
# Shape: (50, 40, 30) means [50][40][30] in C notation
```

```cpp
// C++ equivalent using ndarray:
ndarray<double> arr;
arr.reshapec(50, 40, 30);  // Match NumPy's shape specification

// Or equivalently:
arr.reshapef(30, 40, 50);  // Fortran-order internally

// To match NumPy's indexing:
arr.c(i, j, k);  // i ∈ [0,50), j ∈ [0,40), k ∈ [0,30)
```

## Implementation Notes

### Storage is Always Fortran-Order

```cpp
std::vector<T> p;  // Data always stored in column-major (Fortran) order
```

The `c` functions are **convenience wrappers** that:
1. Reverse dimension specifications (`reshapec`)
2. Reverse dimension queries (`dimc`, `shapec`)
3. Reverse index order (`c()` element access)

### Stride Calculation

Strides are always computed for column-major:

```cpp
void reshapef(const std::vector<size_t>& dims_) {
  s[0] = 1;
  s[1] = dims[0];
  s[2] = dims[0] * dims[1];
  // ...
}
```

The `c()` accessor computes indices by reversing:

```cpp
T& c(size_t i0, size_t i1, size_t i2) {
  return p[i2 + i1*s[1] + i0*s[2]];  // Indices reversed!
}
```

## Best Practices

1. **Be Consistent**: Pick one convention and stick to it
   ```cpp
   // GOOD: All Fortran-style
   arr.reshapef(nx, ny, nz);
   size_t n = arr.dimf(0);
   arr.f(x, y, z);

   // MIXED (confusing):
   arr.reshapef(nx, ny, nz);
   size_t n = arr.dimc(0);  // Returns nz, not nx!
   ```

2. **Default to Fortran-style**: It matches internal storage
   ```cpp
   // Preferred:
   arr.reshapef(3, 4, 5);
   arr.dimf(i);
   arr.f(i, j, k);
   ```

3. **Use C-style only when interfacing with C code**
   ```cpp
   // When matching C array declarations:
   // double c_array[5][4][3];
   arr.reshapec(5, 4, 3);
   arr.c(i, j, k);
   ```

4. **Document your choice**
   ```cpp
   // Dimensions: [nx=100, ny=200, nz=50] (Fortran-order)
   arr.reshapef(100, 200, 50);
   ```

## API Summary

| Operation | Fortran-style (f) | C-style (c) |
|-----------|-------------------|-------------|
| Reshape | `reshapef(nx, ny, nz)` | `reshapec(nz, ny, nx)` |
| Get dimension | `dimf(i)` returns `dims[i]` | `dimc(i)` returns `dims[n-i-1]` |
| Get shape | `shapef()` returns `dims` | `shapec()` returns reversed `dims` |
| Element access | `f(i, j, k)` | `c(k, j, i)` |
| Memory order | Column-major | Column-major (same!) |

## See Also

- [ARRAY_INDEXING.md](ARRAY_INDEXING.md) - Details on f() vs c() element access
- [MULTICOMPONENT_ARRAYS.md](MULTICOMPONENT_ARRAYS.md) - Component dimensions
- [TIME_DIMENSION.md](TIME_DIMENSION.md) - Time dimension handling
