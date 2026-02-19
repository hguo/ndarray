# Multicomponent Arrays

This document explains how to work with multicomponent (vector/tensor) fields in ndarray.

## Overview

Multicomponent arrays allow you to store vector fields, tensor fields, or other data with multiple values at each spatial point. Examples include:

- **Velocity fields**: 2 or 3 components (vx, vy, vz) at each spatial location
- **RGB images**: 3 components (R, G, B) at each pixel
- **Stress tensors**: 9 components (3×3 matrix) at each point
- **Jacobian matrices**: n×m components at each point

## Core Concept

The `n_component_dims` member (accessed via `multicomponents()`) indicates how many **leading dimensions** represent components rather than spatial/temporal dimensions.

```cpp
size_t multicomponents() const;  // Returns n_component_dims (0, 1, or 2)
```

**Key principle**: Component dimensions are **always the first dimensions** in the array shape.

## Memory Layout

Understanding memory layout is crucial for performance:

```cpp
// For a velocity field with shape [3, 100, 200]:
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200);  // [ncomp, nx, ny]
velocity.set_multicomponents(1);

// Memory layout (Fortran-order: first index varies fastest):
// [vx(0,0), vy(0,0), vz(0,0), vx(1,0), vy(1,0), vz(1,0), ...]
//  ^---components at (0,0)--^  ^---components at (1,0)--^
```

**All components at a single spatial location are contiguous in memory**, providing excellent cache locality.

## Dimension Interpretation

The interpretation of array dimensions depends on `n_component_dims`:

### Scalar Field (n_component_dims = 0)

All dimensions are spatial/temporal:

```cpp
ftk::ndarray<double> temperature;
temperature.reshapef(100, 200, 50);  // 100 × 200 × 50 grid
temperature.set_multicomponents(0);   // Scalar field (default)

// Direct spatial indexing with f() (Fortran-order)
double T = temperature.f(x, y, z);  // x ∈ [0,100), y ∈ [0,200), z ∈ [0,50)

// Or with c() (C-order, NumPy-style)
double T = temperature.c(z, y, x);  // z ∈ [0,50), y ∈ [0,200), x ∈ [0,100)
```

**Shape**: `[nx, ny, nz]`
**Interpretation**: All dimensions are spatial
**Total elements**: `nx × ny × nz`

### Vector Field (n_component_dims = 1)

The **first dimension** contains components:

```cpp
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200, 50);  // 3-component velocity field
velocity.set_multicomponents(1);      // First dimension is components

// Access components using f() (Fortran-order: component index first)
float vx = velocity.f(0, x, y, z);  // x-component
float vy = velocity.f(1, x, y, z);  // y-component
float vz = velocity.f(2, x, y, z);  // z-component

// Access using c() (C-order: component index last in memory, but first in call)
// Note: c() reverses ALL indices, including component dimension
float vx = velocity.c(0, z, y, x);  // Same as f(x, y, z, 0) - rarely used

// Common usage: iterate over all spatial points, accessing all components
for (size_t k = 0; k < 50; k++) {
  for (size_t j = 0; j < 200; j++) {
    for (size_t i = 0; i < 100; i++) {  // Inner loop: i varies fastest
      // All 3 components at (i,j,k) are contiguous in memory
      float vx = velocity.f(0, i, j, k);
      float vy = velocity.f(1, i, j, k);
      float vz = velocity.f(2, i, j, k);
      float speed = sqrt(vx*vx + vy*vy + vz*vz);
    }
  }
}
```

**Shape**: `[nc, nx, ny, nz]` where `nc` = number of components
**Interpretation**: `dims[0]` = components, rest are spatial
**Total elements**: `nc × nx × ny × nz`
**Spatial dimensions**: `nd() - multicomponents()` = 4 - 1 = 3

#### Total Components

```cpp
size_t ncomponents() const {
  size_t rtn = 1;
  for (size_t i = 0; i < ncd; i++)
    rtn *= dims[i];
  return rtn;
}
```

For a 3-component velocity field: `ncomponents() = dims[0] = 3`

### Tensor Field (n_component_dims = 2)

The **first two dimensions** contain components:

```cpp
ftk::ndarray<double> jacobian;
jacobian.reshapef(3, 3, 100, 200);  // 3×3 Jacobian matrix at each point
jacobian.set_multicomponents(2);     // First two dimensions are components

// Access tensor elements using f() (Fortran-order)
// f(row, col, x, y) for a 2D spatial field
double J_xy = jacobian.f(0, 1, x, y);  // J[0,1] = ∂x/∂y at position (x, y)
double J_xx = jacobian.f(0, 0, x, y);  // J[0,0] = ∂x/∂x at position (x, y)

// Memory layout: all 9 components of the 3×3 matrix at (i,j) are contiguous
// [J00(0,0), J10(0,0), J20(0,0), J01(0,0), ..., J22(0,0), J00(1,0), ...]

// Example: compute determinant at each spatial point
for (size_t j = 0; j < 200; j++) {
  for (size_t i = 0; i < 100; i++) {
    // Extract 3×3 matrix at this point (all components are nearby in memory)
    double J[3][3];
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        J[row][col] = jacobian.f(row, col, i, j);
      }
    }
    double det = compute_determinant(J);
  }
}
```

**Shape**: `[nc1, nc2, nx, ny]`
**Interpretation**: `dims[0]` and `dims[1]` are component dimensions, rest are spatial
**Total elements**: `nc1 × nc2 × nx × ny`
**Total components per point**: `ncomponents() = dims[0] × dims[1] = 9`
**Spatial dimensions**: `nd() - multicomponents()` = 4 - 2 = 2

## API Functions

### Query Functions

```cpp
// Get number of component dimensions (0, 1, or 2)
size_t multicomponents() const;  // Returns n_component_dims

// Get total number of components (product of first n_component_dims dimensions)
size_t ncomponents() const;

// Get total number of dimensions (including both component and spatial)
size_t nd() const;

// Examples:
// Scalar 3D field:  nd()=3, multicomponents()=0, ncomponents()=1
// Vector 3D field:  nd()=4, multicomponents()=1, ncomponents()=3
// Tensor 2D field:  nd()=4, multicomponents()=2, ncomponents()=9
```

### Setting Component Dimensions

```cpp
// Set number of component dimensions
void set_multicomponents(size_t c = 1);

ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200);
velocity.set_multicomponents(1);  // Mark first dimension as components
```

**IMPORTANT**: Always call `set_multicomponents()` after `reshapef()` when creating multicomponent arrays.

### Converting Scalar to Multicomponent

```cpp
// Convert scalar to vector by adding a dimension of size 1 at front
void make_multicomponents();

// Example: [100, 200] → [1, 100, 200]
ftk::ndarray<double> scalar;
scalar.reshapef(100, 200);        // Shape: [100, 200]
scalar.make_multicomponents();    // Shape: [1, 100, 200], multicomponents()=1

// Useful for treating scalar fields uniformly with vector fields
```

### VTK Integration

VTK uses component information differently:

```cpp
// Writing to VTK
vtkSmartPointer<vtkImageData> to_vtk_image_data() const {
  if (ncd) { // multicomponent
    if (nd() == 3)
      d->SetDimensions(shapef(1), shapef(2), 1);     // 2D: skip component dim
    else
      d->SetDimensions(shapef(1), shapef(2), shapef(3)); // 3D: skip component dim
  } else {
    // Scalar: use all dimensions
    if (nd() == 2) d->SetDimensions(shapef(0), shapef(1), 1);
    else d->SetDimensions(shapef(0), shapef(1), shapef(2));
  }
}
```

**Key insight**: When `ncd > 0`, VTK dimensions skip the component dimensions.

### Reading from VTK

```cpp
void from_vtk_regular_data(vtkSmartPointer<VTK_REGULAR_DATA> d) {
  vtkSmartPointer<vtkDataArray> da = d->GetPointData()->GetArray(...);
  const int nc = da->GetNumberOfComponents();

  if (nc > 1) {
    reshapef(nc, vdims[0], vdims[1]);
    set_multicomponents(1);  // Mark as multicomponent
  } else {
    reshapef(vdims[0], vdims[1]);
    set_multicomponents(0);  // Scalar field
  }
}
```

## Practical Examples

### Example 1: 2D Velocity Field

```cpp
ftk::ndarray<float> velocity;
velocity.reshapef(2, 512, 512);  // 2D velocity (vx, vy) on 512×512 grid
velocity.set_multicomponents(1);

// Fill velocity components using f() (Fortran-order: component first)
for (size_t y = 0; y < 512; y++) {
  for (size_t x = 0; x < 512; x++) {  // Inner loop: x varies fastest
    velocity.f(0, x, y) = compute_vx(x, y);  // x-component
    velocity.f(1, x, y) = compute_vy(x, y);  // y-component
  }
}

// Compute velocity magnitude at each point
ftk::ndarray<float> speed;
speed.reshapef(512, 512);
speed.set_multicomponents(0);  // Scalar field

for (size_t y = 0; y < 512; y++) {
  for (size_t x = 0; x < 512; x++) {
    float vx = velocity.f(0, x, y);
    float vy = velocity.f(1, x, y);
    speed.f(x, y) = sqrt(vx*vx + vy*vy);
  }
}

// Query array properties
std::cout << "nd() = " << velocity.nd() << std::endl;                    // 3
std::cout << "multicomponents() = " << velocity.multicomponents() << std::endl;  // 1
std::cout << "ncomponents() = " << velocity.ncomponents() << std::endl;  // 2
std::cout << "Spatial dims: " << velocity.shapef(1) << " × "
          << velocity.shapef(2) << std::endl;                             // 512 × 512
std::cout << "Total elements: " << velocity.size() << std::endl;         // 524288 (2×512×512)
```

### Example 2: RGB Image

```cpp
ftk::ndarray<uint8_t> rgb_image;
rgb_image.reshapef(3, 1920, 1080);  // 1920×1080 RGB image
rgb_image.set_multicomponents(1);

// Set a red pixel at (100, 200)
rgb_image.f(0, 100, 200) = 255;  // R
rgb_image.f(1, 100, 200) = 0;    // G
rgb_image.f(2, 100, 200) = 0;    // B

// Convert to grayscale
ftk::ndarray<uint8_t> gray;
gray.reshapef(1920, 1080);
gray.set_multicomponents(0);

for (size_t y = 0; y < 1080; y++) {
  for (size_t x = 0; x < 1920; x++) {
    float r = rgb_image.f(0, x, y);
    float g = rgb_image.f(1, x, y);
    float b = rgb_image.f(2, x, y);
    gray.f(x, y) = static_cast<uint8_t>(0.299*r + 0.587*g + 0.114*b);
  }
}
```

### Example 3: 3D Stress Tensor

```cpp
ftk::ndarray<double> stress;
stress.reshapef(3, 3, 64, 64, 64);  // 3×3 stress tensor at each point in 64³ grid
stress.set_multicomponents(2);

// Fill tensor components
for (size_t z = 0; z < 64; z++) {
  for (size_t y = 0; y < 64; y++) {
    for (size_t x = 0; x < 64; x++) {  // Inner loop: x varies fastest
      // Diagonal components
      stress.f(0, 0, x, y, z) = sigma_xx(x, y, z);
      stress.f(1, 1, x, y, z) = sigma_yy(x, y, z);
      stress.f(2, 2, x, y, z) = sigma_zz(x, y, z);

      // Off-diagonal (symmetric tensor: σ_ij = σ_ji)
      double sxy = sigma_xy(x, y, z);
      stress.f(0, 1, x, y, z) = sxy;
      stress.f(1, 0, x, y, z) = sxy;

      double sxz = sigma_xz(x, y, z);
      stress.f(0, 2, x, y, z) = sxz;
      stress.f(2, 0, x, y, z) = sxz;

      double syz = sigma_yz(x, y, z);
      stress.f(1, 2, x, y, z) = syz;
      stress.f(2, 1, x, y, z) = syz;
    }
  }
}

// Compute von Mises stress at each point
ftk::ndarray<double> von_mises;
von_mises.reshapef(64, 64, 64);
von_mises.set_multicomponents(0);

for (size_t z = 0; z < 64; z++) {
  for (size_t y = 0; y < 64; y++) {
    for (size_t x = 0; x < 64; x++) {
      double sxx = stress.f(0, 0, x, y, z);
      double syy = stress.f(1, 1, x, y, z);
      double szz = stress.f(2, 2, x, y, z);
      double sxy = stress.f(0, 1, x, y, z);
      double sxz = stress.f(0, 2, x, y, z);
      double syz = stress.f(1, 2, x, y, z);

      von_mises.f(x, y, z) = compute_von_mises(sxx, syy, szz, sxy, sxz, syz);
    }
  }
}

std::cout << "nd() = " << stress.nd() << std::endl;                    // 5
std::cout << "multicomponents() = " << stress.multicomponents() << std::endl;  // 2
std::cout << "ncomponents() = " << stress.ncomponents() << std::endl;  // 9
std::cout << "Total elements: " << stress.size() << std::endl;         // 2359296 (9×64×64×64)
```

### Example 4: Extracting Single Component

```cpp
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200, 50);  // 3D velocity field
velocity.set_multicomponents(1);

// Extract x-component as a separate scalar field
ftk::ndarray<float> vx;
vx.reshapef(100, 200, 50);
vx.set_multicomponents(0);

for (size_t k = 0; k < 50; k++) {
  for (size_t j = 0; j < 200; j++) {
    for (size_t i = 0; i < 100; i++) {
      vx.f(i, j, k) = velocity.f(0, i, j, k);  // Extract component 0
    }
  }
}

// Or extract all components into separate arrays
std::vector<ftk::ndarray<float>> components(3);
for (int c = 0; c < 3; c++) {
  components[c].reshapef(100, 200, 50);
  components[c].set_multicomponents(0);

  for (size_t k = 0; k < 50; k++) {
    for (size_t j = 0; j < 200; j++) {
      for (size_t i = 0; i < 100; i++) {
        components[c].f(i, j, k) = velocity.f(c, i, j, k);
      }
    }
  }
}
```

## Design Rationale

### Why Component-First Ordering?

Components are placed in the **first dimensions** (varying fastest in memory) for several important reasons:

#### 1. Cache Locality

All components at a single spatial point are **contiguous in memory**, providing optimal cache performance:

```cpp
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200);  // Shape: [3, 100, 200]
velocity.set_multicomponents(1);

// Memory layout: [vx(0,0), vy(0,0), vz(0,0), vx(1,0), vy(1,0), vz(1,0), ...]
//                 ^---- contiguous ------^  ^---- contiguous ------^

// GOOD: Access all components at one point (sequential memory access)
float vx = velocity.f(0, x, y);  // Loads cache line with all 3 components
float vy = velocity.f(1, x, y);  // Already in cache!
float vz = velocity.f(2, x, y);  // Already in cache!

// BAD (if components were last): Each component would be far apart in memory
// [vx(0,0), vx(1,0), ..., vx(99,199), vy(0,0), vy(1,0), ..., vz(99,199)]
//  ^----- 20000 elements apart ----^  ^----- 20000 elements apart ----^
```

#### 2. VTK Compatibility

VTK's `vtkDataArray` uses component-first storage, making conversion zero-copy in many cases.

#### 3. Fortran Column-Major Convention

First index varies fastest in Fortran-order (column-major), naturally aligning with scientific computing conventions.

#### 4. Natural Iteration Pattern

Outer loops over space, inner operations on components at each point:

```cpp
for (size_t y = 0; y < ny; y++) {
  for (size_t x = 0; x < nx; x++) {
    // All components at (x,y) accessed together
    process_vector(velocity.f(0,x,y), velocity.f(1,x,y), velocity.f(2,x,y));
  }
}
```

### Relationship to nd()

The total number of dimensions (`nd()`) includes both component and spatial dimensions:

```cpp
nd() = ncd + spatial_dimensions
```

**Examples:**
- Scalar 3D field: `nd() = 3`, `ncd = 0` → 3 spatial dims
- Vector 3D field: `nd() = 4`, `ncd = 1` → 1 component dim + 3 spatial dims
- Tensor 2D field: `nd() = 4`, `ncd = 2` → 2 component dims + 2 spatial dims

## Time-Varying Fields

The `tv` flag indicates if the **last dimension** is time:

```cpp
ndarray<float> velocity_timeseries;
velocity_timeseries.reshapef(3, 100, 200, 50, 100);  // 3D velocity over 100 timesteps
velocity_timeseries.set_multicomponents(1);
velocity_timeseries.set_has_time(true);

// Access
float vx_at_t50 = velocity_timeseries.f(0, x, y, z, 50);

// Slice by time
ndarray<float> velocity_t50 = velocity_timeseries.slice_time(50);
// velocity_t50.shapef() = [3, 100, 200, 50]
// velocity_t50.has_time() = false
```

**Dimension ordering with time:**
```
[component_dims... , spatial_dims... , time_dim]
 <------ ncd ------>                    <-- tv -->
```

## Indexing with f() and c()

### Using f() (Fortran-order)

**Recommended for multicomponent arrays**. Component indices come first:

```cpp
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200);  // [nc=3, nx=100, ny=200]
velocity.set_multicomponents(1);

// f(component, x, y): component index first, then spatial indices
float vx = velocity.f(0, 50, 100);  // c=0, x=50, y=100
```

### Using c() (C-order)

**Rarely used with multicomponent arrays**. Remember that `c()` reverses ALL indices:

```cpp
// c() reverses: c(y, x, component)
// This is confusing and not recommended for multicomponent arrays
float vx = velocity.c(100, 50, 0);  // Same as f(0, 50, 100)
```

**Recommendation**: Use `f()` consistently with multicomponent arrays.

## Best Practices

### 1. Always Set Multicomponents Explicitly

```cpp
// GOOD
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200);
velocity.set_multicomponents(1);  // Explicitly mark as multicomponent

// BAD - Missing set_multicomponents()
ftk::ndarray<float> velocity;
velocity.reshapef(3, 100, 200);
// multicomponents() == 0 by default! Will be treated as 3D scalar field
```

### 2. Use Descriptive Names for Dimensions

```cpp
// GOOD: Clear what each dimension represents
const size_t ncomp = 3;
const size_t nx = 100;
const size_t ny = 200;
velocity.reshapef(ncomp, nx, ny);
velocity.set_multicomponents(1);

// BAD: Unclear dimensions
velocity.reshapef(3, 100, 200);  // What does each number mean?
```

### 3. Query Properties for Generic Code

```cpp
void process_field(const ftk::ndarray<float>& field) {
  size_t ncomp_dims = field.multicomponents();
  size_t total_comps = field.ncomponents();

  if (ncomp_dims == 0) {
    // Scalar field: field.f(x, y, z)
  } else if (ncomp_dims == 1) {
    // Vector field: field.f(c, x, y, z)
    size_t nc = field.shapef(0);  // Number of components
  } else if (ncomp_dims == 2) {
    // Tensor field: field.f(i, j, x, y, z)
    size_t ni = field.shapef(0);
    size_t nj = field.shapef(1);
  }
}
```

### 4. Optimize Loops for Cache Performance

```cpp
// GOOD: Inner loop over spatial indices (component loop inside spatial loops)
for (size_t y = 0; y < ny; y++) {
  for (size_t x = 0; x < nx; x++) {
    for (size_t c = 0; c < 3; c++) {  // Components are contiguous
      process(velocity.f(c, x, y));
    }
  }
}

// ALSO GOOD: Unroll component loop for small fixed sizes
for (size_t y = 0; y < ny; y++) {
  for (size_t x = 0; x < nx; x++) {
    float vx = velocity.f(0, x, y);
    float vy = velocity.f(1, x, y);
    float vz = velocity.f(2, x, y);
    process(vx, vy, vz);
  }
}

// BAD: Outer loop over components (poor cache utilization)
for (size_t c = 0; c < 3; c++) {
  for (size_t y = 0; y < ny; y++) {
    for (size_t x = 0; x < nx; x++) {
      process(velocity.f(c, x, y));
    }
  }
}
```

### 5. Document Component Interpretation

```cpp
// GOOD: Document what each component represents
ftk::ndarray<float> velocity;
velocity.reshapef(3, nx, ny, nz);  // [vx, vy, vz] components
velocity.set_multicomponents(1);
// Component 0: x-velocity (vx)
// Component 1: y-velocity (vy)
// Component 2: z-velocity (vz)

ftk::ndarray<double> stress;
stress.reshapef(3, 3, nx, ny);  // [σ_ij] where i,j ∈ {0,1,2} = {x,y,z}
stress.set_multicomponents(2);
// Component (i,j): stress tensor component σ_ij
```

## Quick Reference

### API Summary

```cpp
// Setting/getting component dimensions
void set_multicomponents(size_t c = 1);     // Set number of component dimensions (0, 1, or 2)
size_t multicomponents() const;              // Get number of component dimensions
size_t ncomponents() const;                  // Get total components (product of component dims)

// Conversion
void make_multicomponents();                 // Convert scalar to vector: [nx,ny] → [1,nx,ny]

// Dimension queries
size_t nd() const;                           // Total dimensions (component + spatial + time)
size_t dimf(size_t i) const;                 // Get dimension i (Fortran-order)
const std::vector<size_t>& shapef() const;   // Get all dimensions (Fortran-order)

// Element access (always use Fortran-order f() with multicomponent arrays)
T& f(...);                                   // Access: f(comp_indices..., spatial_indices...)
const T& f(...) const;
```

### Common Patterns

| Array Type | Shape | multicomponents() | ncomponents() | Access Pattern |
|------------|-------|-------------------|---------------|----------------|
| Scalar 2D | `[nx, ny]` | 0 | 1 | `f(x, y)` |
| Scalar 3D | `[nx, ny, nz]` | 0 | 1 | `f(x, y, z)` |
| Vector 2D | `[nc, nx, ny]` | 1 | nc | `f(c, x, y)` |
| Vector 3D | `[nc, nx, ny, nz]` | 1 | nc | `f(c, x, y, z)` |
| Tensor 2D | `[n1, n2, nx, ny]` | 2 | n1×n2 | `f(i, j, x, y)` |
| Tensor 3D | `[n1, n2, nx, ny, nz]` | 2 | n1×n2 | `f(i, j, x, y, z)` |

### Dimension Calculation

```cpp
// Relationship between dimensions
nd() = multicomponents() + spatial_dimensions [+ time_dimension]

// Examples:
// Scalar 3D:      nd()=3, multicomponents()=0  →  3 spatial dims
// Vector 3D:      nd()=4, multicomponents()=1  →  3 spatial dims
// Tensor 2D:      nd()=4, multicomponents()=2  →  2 spatial dims
// Vector 3D+time: nd()=5, multicomponents()=1  →  3 spatial + 1 time dim
```

## Common Pitfalls

### Pitfall 1: Forgetting to set ncd

```cpp
ndarray<float> vel;
vel.reshapef(3, 100, 200);
// vel.multicomponents() == 0  <-- WRONG! Treated as 3D scalar field
```

**Fix:**
```cpp
vel.reshapef(3, 100, 200);
vel.set_multicomponents(1);  // Correct!
```

### Pitfall 2: Wrong indexing order

```cpp
ndarray<float> vel;
vel.reshapef(3, 100, 200);
vel.set_multicomponents(1);

// WRONG: treats x as component
float bad = vel.f(x, y, c);

// CORRECT: component first
float good = vel.f(c, x, y);
```

### Pitfall 3: Slicing doesn't adjust ncd

```cpp
// Component dimensions are preserved in slices
// You may need to manually adjust ncd if extracting a single component
```

## See Also

- [ARRAY_ACCESS.md](ARRAY_ACCESS.md) - Detailed guide to f() and c() element access
- [FORTRAN_C_CONVENTIONS.md](FORTRAN_C_CONVENTIONS.md) - Understanding F/C ordering
- [TIME_DIMENSION.md](TIME_DIMENSION.md) - Handling time-varying multicomponent fields
- [VTK_TESTS.md](VTK_TESTS.md) - VTK integration examples

---

**Last Updated**: 2026-02-19
