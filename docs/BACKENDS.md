# Eigen and xtensor Backend Integration

## Overview

ndarray provides optional integration with two popular C++ tensor libraries:

- **Eigen** - Linear algebra library for matrices and vectors
- **xtensor** - Multi-dimensional array library with NumPy-like API

These backends enable seamless interoperability: read data with ndarray, process with Eigen/xtensor, write back with ndarray.

## Building with Backend Support

### Eigen Backend

**Option 1: CMake detects Eigen automatically**
```bash
cmake -DNDARRAY_USE_EIGEN=AUTO ..
```

**Option 2: Require Eigen (fails if not found)**
```bash
cmake -DNDARRAY_USE_EIGEN=TRUE -DEigen3_DIR=/path/to/eigen3/cmake ..
```

**Option 3: Disable Eigen**
```bash
cmake -DNDARRAY_USE_EIGEN=FALSE ..
```

### xtensor Backend

**Option 1: CMake detects xtensor automatically**
```bash
cmake -DNDARRAY_USE_XTENSOR=AUTO ..
```

**Option 2: Require xtensor (fails if not found)**
```bash
cmake -DNDARRAY_USE_XTENSOR=TRUE -Dxtensor_DIR=/path/to/xtensor/cmake ..
```

**Option 3: Disable xtensor**
```bash
cmake -DNDARRAY_USE_XTENSOR=FALSE ..
```

### Combined Example

```bash
cmake \
  -DNDARRAY_USE_EIGEN=TRUE \
  -DNDARRAY_USE_XTENSOR=TRUE \
  -DNDARRAY_USE_NETCDF=TRUE \
  -DNDARRAY_USE_HDF5=TRUE \
  ..
```

## Eigen Backend Usage

### Include Header

```cpp
#include <ndarray/ndarray_eigen.hh>
```

### Convert ndarray to Eigen

**2D arrays (matrices)**
```cpp
ftk::ndarray<double> arr;
arr.read_netcdf("data.nc", "temperature");  // 100 x 200 array

// Convert to Eigen matrix
auto mat = ftk::ndarray_to_eigen(arr);

// Now use Eigen operations
Eigen::VectorXd col_means = mat.colwise().mean();
Eigen::MatrixXd normalized = mat.rowwise() - col_means.transpose();
```

**1D arrays (vectors)**
```cpp
ftk::ndarray<double> arr;
arr.reshapef(1000);
// ... fill with data ...

// Convert to Eigen vector
auto vec = ftk::ndarray_to_eigen_vector(arr);

// Eigen operations
double norm = vec.norm();
double mean = vec.mean();
```

### Convert Eigen to ndarray

**Matrices**
```cpp
// Create Eigen matrix
Eigen::MatrixXd mat(100, 50);
mat.setRandom();

// Convert to ndarray
auto arr = ftk::eigen_to_ndarray(mat);

// Write to file
arr.write_netcdf("output.nc", "result");
```

**Vectors**
```cpp
Eigen::VectorXd vec(1000);
vec.setLinSpaced(1000, 0.0, 10.0);

auto arr = ftk::eigen_vector_to_ndarray(vec);
arr.write_h5("output.h5", "linspace");
```

### Complete Workflow Example

```cpp
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_eigen.hh>
#include <Eigen/Dense>

int main() {
  // 1. Read data with ndarray
  ftk::ndarray<double> temperature;
  temperature.read_netcdf("climate.nc", "temperature");  // 365 x 100 x 100

  // Extract 2D slice (day 0)
  ftk::ndarray<double> day0;
  day0.reshapef(100, 100);
  for (size_t i = 0; i < 100; i++) {
    for (size_t j = 0; j < 100; j++) {
      day0.at(i, j) = temperature.at(0, i, j);
    }
  }

  // 2. Convert to Eigen for linear algebra
  auto mat = ftk::ndarray_to_eigen(day0);

  // 3. Compute eigenvalues (e.g., spatial correlation analysis)
  Eigen::MatrixXd cov = (mat.adjoint() * mat) / mat.rows();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
  Eigen::VectorXd eigenvalues = solver.eigenvalues();

  // 4. Convert result back to ndarray
  auto result = ftk::eigen_vector_to_ndarray(eigenvalues);

  // 5. Write result
  result.write_netcdf("eigenvalues.nc", "spatial_modes");

  return 0;
}
```

## xtensor Backend Usage

### Include Header

```cpp
#include <ndarray/ndarray_xtensor.hh>
#include <xtensor/xarray.hpp>
```

### Convert ndarray to xtensor

**Any dimensionality (1D-4D)**
```cpp
ftk::ndarray<float> arr;
arr.read_netcdf("velocity.nc", "u");  // 100 x 200 x 50 x 3

// Convert to xtensor (makes a copy)
auto xarr = ftk::ndarray_to_xtensor(arr);

// xtensor operations (NumPy-like API)
auto mean = xt::mean(xarr);
auto stddev = xt::stddev(xarr);
auto normalized = (xarr - mean) / stddev;
```

### Zero-Copy Views

For large arrays, avoid copying by using views:

```cpp
ftk::ndarray<double> arr;
arr.read_netcdf("large.nc", "field");  // 1000 x 1000 x 1000

// Create zero-copy view (no allocation!)
auto view = ftk::ndarray_to_xtensor_view(arr);

// Read-only operations on view
auto slice = xt::view(view, xt::all(), 500, xt::all());
double max_val = xt::amax(slice)();

// Warning: view is invalid if arr is destroyed or reshaped
```

### Convert xtensor to ndarray

```cpp
// Create xtensor array
xt::xarray<double> xarr = xt::random::randn<double>({100, 200, 50});

// Convert to ndarray
auto arr = ftk::xtensor_to_ndarray(xarr);

// Write to file
arr.write_bp("output.bp", "random_field", 0);
```

### Complete Workflow Example

```cpp
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_xtensor.hh>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

int main() {
  // 1. Read time-varying 3D data
  ftk::ndarray<float> velocity;
  velocity.read_netcdf("flow.nc", "velocity");  // time x z x y x x x 3

  // 2. Convert to xtensor
  auto xvel = ftk::ndarray_to_xtensor(velocity);

  // 3. Compute vorticity using xtensor operations
  // Extract velocity components
  auto u = xt::view(xvel, xt::all(), xt::all(), xt::all(), xt::all(), 0);
  auto v = xt::view(xvel, xt::all(), xt::all(), xt::all(), xt::all(), 1);
  auto w = xt::view(xvel, xt::all(), xt::all(), xt::all(), xt::all(), 2);

  // Compute spatial derivatives (simplified - use proper finite differences)
  auto vorticity_x = xt::diff(w, 2) - xt::diff(v, 3);  // dw/dy - dv/dz

  // 4. Convert result back to ndarray
  auto result = ftk::xtensor_to_ndarray(vorticity_x);

  // 5. Write result
  result.write_netcdf("vorticity.nc", "vorticity_x");

  return 0;
}
```

## Performance Considerations

### Eigen Backend

- **Conversion cost**: O(n) copy for matrix/vector conversions
- **Use case**: Best for linear algebra operations (SVD, eigendecomposition, matrix factorization)
- **Memory**: Creates new allocations during conversion

### xtensor Backend

- **Conversion cost**: O(n) copy for `ndarray_to_xtensor()` and `xtensor_to_ndarray()`
- **Zero-copy views**: `ndarray_to_xtensor_view()` has O(1) cost
- **Use case**: Best for element-wise operations, broadcasting, slicing
- **Memory**: Views avoid allocation; conversions create copies

### When to Use Each Backend

| Operation | Recommended Backend |
|-----------|-------------------|
| Matrix multiplication | Eigen |
| Eigenvalue decomposition | Eigen |
| SVD, QR, Cholesky | Eigen |
| Element-wise operations | xtensor |
| Broadcasting | xtensor |
| Slicing/indexing | xtensor (views) |
| NumPy-like API | xtensor |
| Large array processing | xtensor (views) |

## Memory Layout

### ndarray
- Default: **Column-major (Fortran order)**
- `dimf(0)` is the fastest-varying dimension

### Eigen
- Default: **Row-major**
- Conversion functions handle layout automatically

### xtensor
- Default: **Row-major (C order)**
- Views use `column_major` layout to match ndarray

**Important**: The conversion functions handle layout differences automatically. You don't need to worry about memory layout unless you're accessing raw pointers.

## Error Handling

All conversion functions throw exceptions on errors:

```cpp
try {
  ftk::ndarray<double> arr;
  arr.reshapef(10, 20, 30);  // 3D array

  // This throws: Eigen requires 2D arrays
  auto mat = ftk::ndarray_to_eigen(arr);

} catch (const std::runtime_error& e) {
  std::cerr << "Error: " << e.what() << std::endl;
}
```

**Common errors**:
- `ndarray_to_eigen()` requires 2D arrays
- `ndarray_to_eigen_vector()` requires 1D arrays
- `ndarray_to_xtensor()` and `xtensor_to_ndarray()` support only 1D-4D arrays

## Limitations

### Eigen Backend
- Only supports 1D and 2D arrays
- No sparse matrix support
- No direct support for multi-component arrays (use slicing)

### xtensor Backend
- Conversion functions limited to 1D-4D arrays
- Views are invalidated if ndarray is reshaped or destroyed
- No automatic broadcasting during conversion

## Building Examples

If you have a program using these backends:

```bash
# With Eigen
g++ -std=c++17 my_program.cpp \
    -I/path/to/ndarray/include \
    -I/path/to/eigen3 \
    -lnetcdf

# With xtensor
g++ -std=c++17 my_program.cpp \
    -I/path/to/ndarray/include \
    -I/path/to/xtensor/include \
    -I/path/to/xtl/include \
    -lnetcdf
```

## See Also

- [Eigen documentation](https://eigen.tuxfamily.org/)
- [xtensor documentation](https://xtensor.readthedocs.io/)
- [ARRAY_ACCESS.md](ARRAY_ACCESS.md) - ndarray dimension and indexing
- [ZERO_COPY_OPTIMIZATION.md](ZERO_COPY_OPTIMIZATION.md) - Performance optimization
