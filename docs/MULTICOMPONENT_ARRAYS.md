# Multicomponent Arrays

This document clarifies how multicomponent arrays work in ndarray.

## Core Concept

The `ncd` member variable (number of component dimensions) indicates how many **leading dimensions** represent components rather than spatial/temporal dimensions.

```cpp
size_t ncd = 0;  // From ndarray_base.hh:282
```

## Dimension Interpretation

The interpretation of array dimensions depends on `ncd`:

### Scalar Field (ncd = 0)

All dimensions are spatial/temporal:

```cpp
ndarray<double> temperature;
temperature.reshapef(100, 200, 50);  // 100 x 200 x 50 grid
temperature.set_multicomponents(0);   // Scalar field

// Direct spatial indexing
double T = temperature.f(x, y, z);
```

**Shape**: `[nx, ny, nz]`
**Interpretation**: All dimensions are spatial

### Vector Field (ncd = 1)

The **first dimension** contains components:

```cpp
ndarray<float> velocity;
velocity.reshapef(3, 100, 200, 50);  // 3-component velocity field
velocity.set_multicomponents(1);      // First dimension is components

// Access components
float vx = velocity.f(0, x, y, z);  // x-component
float vy = velocity.f(1, x, y, z);  // y-component
float vz = velocity.f(2, x, y, z);  // z-component
```

**Shape**: `[nc, nx, ny, nz]` where `nc` = number of components
**Interpretation**: `dims[0]` = components, rest are spatial

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

### Tensor Field (ncd = 2)

The **first two dimensions** contain components:

```cpp
ndarray<double> jacobian;
jacobian.reshapef(3, 3, 100, 200);  // 3x3 Jacobian at each point
jacobian.set_multicomponents(2);     // First two dimensions are components

// Access tensor elements
double J_xy = jacobian.f(0, 1, x, y);  // ∂x/∂y component
```

**Shape**: `[nc1, nc2, nx, ny]`
**Interpretation**: `dims[0]` and `dims[1]` are component dimensions, rest are spatial

#### Total Components

For a 3x3 tensor: `ncomponents() = dims[0] * dims[1] = 9`

## Common Operations

### Converting Scalar to Multicomponent

The `make_multicomponents()` function adds a component dimension:

```cpp
void make_multicomponents() {
  std::vector<size_t> s = shapef();
  s.insert(s.begin(), 1);  // Insert dimension 1 at front
  reshapef(s);
  set_multicomponents(1);
}
```

**Example:**
```cpp
ndarray<double> scalar;
scalar.reshapef(100, 200);        // Shape: [100, 200]
scalar.make_multicomponents();    // Shape: [1, 100, 200], ncd=1
```

This is useful when you need to treat a scalar field uniformly with vector fields.

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

## Examples

### Example 1: 2D Velocity Field

```cpp
ndarray<float> velocity;
velocity.reshapef(2, 512, 512);  // 2D velocity (vx, vy) on 512x512 grid
velocity.set_multicomponents(1);

// Fill velocity components
for (size_t y = 0; y < 512; y++) {
  for (size_t x = 0; x < 512; x++) {
    velocity.f(0, x, y) = compute_vx(x, y);  // x-component
    velocity.f(1, x, y) = compute_vy(x, y);  // y-component
  }
}

// Query
std::cout << "nd() = " << velocity.nd() << std::endl;              // 3
std::cout << "ncd = " << velocity.multicomponents() << std::endl;  // 1
std::cout << "ncomponents() = " << velocity.ncomponents() << std::endl; // 2
std::cout << "Spatial dims: " << velocity.shapef(1) << " x " << velocity.shapef(2) << std::endl; // 512 x 512
```

### Example 2: 3D Stress Tensor

```cpp
ndarray<double> stress;
stress.reshapef(3, 3, 64, 64, 64);  // 3x3 symmetric stress tensor
stress.set_multicomponents(2);

// Fill diagonal components
for (size_t z = 0; z < 64; z++) {
  for (size_t y = 0; y < 64; y++) {
    for (size_t x = 0; x < 64; x++) {
      stress.f(0, 0, x, y, z) = sigma_xx(x, y, z);
      stress.f(1, 1, x, y, z) = sigma_yy(x, y, z);
      stress.f(2, 2, x, y, z) = sigma_zz(x, y, z);
      stress.f(0, 1, x, y, z) = sigma_xy(x, y, z);
      // ... etc
    }
  }
}

std::cout << "nd() = " << stress.nd() << std::endl;              // 5
std::cout << "ncd = " << stress.multicomponents() << std::endl;  // 2
std::cout << "ncomponents() = " << stress.ncomponents() << std::endl; // 9
```

### Example 3: Slicing Multicomponent Arrays

```cpp
ndarray<float> velocity;
velocity.reshapef(3, 100, 200, 50);  // 3D velocity field
velocity.set_multicomponents(1);

// Slice in space (preserves all components)
lattice spatial_region({10, 20, 0}, {50, 100, 50});
ndarray<float> sliced = velocity.slice(spatial_region);
// sliced.shapef() = [3, 50, 100, 50]
// sliced.multicomponents() = 1

// Note: Component dimensions are preserved during slicing
```

## Design Rationale

### Why Component-First Ordering?

Components are placed in the **first dimensions** for several reasons:

1. **Cache Locality**: Accessing all components at a point is cache-friendly
   ```cpp
   // All components of one point are contiguous in memory
   for (int c = 0; c < 3; c++)
     process(velocity.f(c, x, y, z));
   ```

2. **VTK Compatibility**: VTK's vtkDataArray uses component-first storage

3. **Fortran Interop**: Column-major indexing naturally places components first

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

## Best Practices

1. **Always set multicomponents explicitly** after reshaping:
   ```cpp
   arr.reshapef(3, 100, 200);
   arr.set_multicomponents(1);  // Don't forget!
   ```

2. **Check ncd before indexing**:
   ```cpp
   if (arr.multicomponents() == 0) {
     // Scalar: arr.f(x, y, z)
   } else if (arr.multicomponents() == 1) {
     // Vector: arr.f(c, x, y, z)
   }
   ```

3. **Use ncomponents() for total component count**:
   ```cpp
   size_t nc = arr.ncomponents();  // Product of first ncd dimensions
   ```

4. **Preserve multicomponent info when copying**:
   ```cpp
   ndarray<T> copy = original;  // ncd is copied
   ```

## API Summary

```cpp
// Setting/getting
void set_multicomponents(size_t c = 1);
size_t multicomponents() const;      // Returns ncd
size_t ncomponents() const;          // Returns product of first ncd dimensions

// Conversion
void make_multicomponents();         // Adds dimension [1, ...] at front

// VTK integration (handles ncd automatically)
void to_vtk_image_data_file(filename, varname);
void from_vtk_image_data(vtkImageData, array_name);
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

- [ARRAY_INDEXING.md](ARRAY_INDEXING.md) - F/C ordering semantics
- [VTK_TESTS.md](VTK_TESTS.md) - VTK integration examples
