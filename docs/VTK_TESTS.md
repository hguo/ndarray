# VTK Unit Tests

This document describes the VTK unit tests for the ndarray library.

## Building with VTK Support

To enable VTK tests, build ndarray with VTK support:

```bash
cmake -DNDARRAY_USE_VTK=ON ..
make
ctest -R vtk
```

## Test Coverage

The VTK test suite (`test_vtk.cpp`) covers the following functionality:

### 1. VTK ImageData File I/O (`test_vtk_image_data_file_io`)

Tests reading and writing VTK image data files (.vti format):
- Creates a 2D scalar field (10x20 array)
- Writes to .vti file with named array
- Reads back and verifies dimensions
- Verifies data integrity (bit-exact match)

**API tested:**
- `to_vtk_image_data_file(filename, varname)`
- `read_vtk_image_data_file(filename, varname)`

### 2. 3D VTK ImageData (`test_vtk_image_data_3d`)

Tests 3D volume data:
- Creates a 3D scalar field (5x6x7 array)
- Writes and reads .vti file
- Verifies 3D indexing with `f(i,j,k)` access
- Tests column-major storage compatibility

**Use case:** Scientific volume data, CT scans, CFD results

### 3. VTK DataArray Conversion (`test_vtk_data_array_conversion`)

Tests conversion between ndarray and vtkDataArray:
- Converts ndarray → vtkDataArray
- Converts vtkDataArray → ndarray
- Verifies round-trip data integrity
- Tests array naming

**API tested:**
- `to_vtk_data_array(varname)`
- `ndarray_base::new_from_vtk_data_array(vtkDataArray)`

**Use case:** Integration with existing VTK pipelines

### 4. Multi-Component Arrays (`test_vtk_multicomponent`)

Tests vector/tensor fields with multiple components:
- Creates 3-component vector field (e.g., velocity: vx, vy, vz)
- Stores as (3, 10, 20) array
- Writes/reads with component preservation
- Verifies each component independently

**Use case:** Velocity fields, displacement vectors, stress tensors

### 5. Direct vtkImageData Conversion (`test_vtk_image_data_direct`)

Tests working with vtkImageData objects directly:
- Creates vtkImageData programmatically
- Converts to ndarray using `new_from_vtk_image_data`
- Verifies data layout matches VTK's expectations
- Tests named array extraction

**Use case:** Integration with VTK visualization pipelines

### 6. Different Data Types (`test_vtk_different_types`)

Tests VTK I/O with various data types:
- `int` arrays
- `float` arrays
- `double` arrays
- Verifies type preservation through I/O

**API tested:**
- Type-specific `vtk_data_type()` methods
- Automatic type detection in `new_from_vtk_*` methods

### 7. to_vtk_image_data Method (`test_vtk_to_vtk_image_data`)

Tests in-memory conversion to vtkImageData:
- Converts ndarray → vtkImageData (no file I/O)
- Verifies VTK dimensions match ndarray dimensions
- Checks scalar data array properties
- Tests indexing conventions (row-major vs column-major)

**Use case:** Real-time visualization, in-memory processing

## VTK Indexing Convention

VTK uses **row-major (C-style)** indexing, while ndarray uses **column-major (Fortran-style)** by default with `f()` access.

When writing to VTK:
```cpp
// ndarray: column-major
arr.f(i, j) = value;  // i varies fastest

// VTK accesses same data as:
vtk_array->GetTuple1(j * ni + i);  // j varies fastest
```

The conversion handles this transparently.

## File Formats

### VTK ImageData (.vti)

XML-based format for structured grids:
- Human-readable XML header
- Binary data encoding
- Supports multi-component arrays
- Parallel I/O with .pvti files

### Compatibility

The tests generate files compatible with:
- ParaView
- VisIt
- VTK-based applications
- Python VTK module

## Example Usage

### Write 2D Scalar Field

```cpp
ftk::ndarray<double> temperature;
temperature.reshapef(100, 200);

// Fill with data...
for (size_t j = 0; j < 200; j++)
  for (size_t i = 0; i < 100; i++)
    temperature.f(i, j) = compute_temperature(i, j);

// Write to VTK file
temperature.to_vtk_image_data_file("temperature.vti", "temperature");
```

### Read from VTK File

```cpp
ftk::ndarray<double> data;
data.read_vtk_image_data_file("simulation_output.vti", "pressure");

// Access data
for (size_t j = 0; j < data.dimf(1); j++)
  for (size_t i = 0; i < data.dimf(0); i++)
    std::cout << data.f(i, j) << " ";
```

### 3D Velocity Field

```cpp
ftk::ndarray<float> velocity;
velocity.reshapef(3, nx, ny, nz);  // 3 components

// Fill velocity components
for (size_t k = 0; k < nz; k++)
  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++) {
      velocity.f(0, i, j, k) = vx(i, j, k);
      velocity.f(1, i, j, k) = vy(i, j, k);
      velocity.f(2, i, j, k) = vz(i, j, k);
    }

velocity.to_vtk_image_data_file("velocity.vti", "velocity");
```

### Integration with VTK Pipeline

```cpp
// Create ndarray from VTK reader
vtkSmartPointer<vtkXMLImageDataReader> reader = ...;
reader->Update();

auto arr = ftk::ndarray_base::new_from_vtk_image_data(
  reader->GetOutput(), "field_name");

// Process with ndarray...

// Convert back to VTK for visualization
auto arr_typed = std::dynamic_pointer_cast<ftk::ndarray<double>>(arr);
vtkSmartPointer<vtkImageData> output = arr_typed->to_vtk_image_data("result");

// Pass to VTK pipeline...
```

## Test Files Generated

When tests run, they create these temporary files:
- `test_output.vti` - 2D scalar field
- `test_3d.vti` - 3D scalar field
- `test_velocity.vti` - 3D vector field
- `test_int.vti` - Integer data
- `test_float.vti` - Float data

These can be inspected with ParaView or other VTK tools.

## Limitations

Current VTK support focuses on **structured image data** (vtkImageData):
- Regular grids with uniform/non-uniform spacing
- 2D and 3D arrays
- Scalar and vector fields

Not currently tested:
- Unstructured grids (vtkUnstructuredGrid) - basic support exists
- Polygonal data (vtkPolyData)
- Rectilinear grids
- Time series

## See Also

- [ARRAY_INDEXING.md](ARRAY_INDEXING.md) - F/C ordering semantics
- [VTK Documentation](https://vtk.org/documentation/)
- [ParaView Guide](https://www.paraview.org/paraview-guide/)
